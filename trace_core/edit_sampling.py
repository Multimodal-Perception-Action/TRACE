"""Trained editor loading and reference-preserving DDIM sampling."""
from contextlib import nullcontext
import hashlib
import numpy as np
import torch
from .editor import EventRevisionEditor
from .editor_base import schedule

def precision(device, kind):
    if kind == 'bf16':
        if torch.device(device).type != 'cuda':
            raise ValueError('BF16 inference requires a CUDA device; use fp32 on CPU')
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return nullcontext()


def load_stats(path):
    with np.load(path, allow_pickle=False) as f:
        mean, std = f['mean'].astype(np.float32), f['std'].astype(np.float32)
    if mean.shape != (40,) or std.shape != (40,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Expected finite mean/std vectors of size 40, with positive std')
    return mean, std


def load_editor(path, device):
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = EventRevisionEditor()
    model.load_state_dict(state['ema'], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def sample_edit(model, raw, lengths, text, scope, request, events, mean, std,
                ids, device='cpu', kind='fp32', steps=50, cfg=1.5, seed=42):
    """Source-only DDIM with per-step reference projection and stable per-case seeds."""
    if not 1 <= steps <= 300:
        raise ValueError('Editing steps must be in [1,300]')
    if raw.shape != (len(lengths),104,40) or scope.shape != raw.shape or scope.dtype != np.bool_:
        raise ValueError('Invalid source/scope contract')
    if request.shape != (len(raw),104,2) or events.shape != (len(raw),26,16) or text.shape != (len(raw),2560):
        raise ValueError('Invalid edit condition shapes')
    if not all(np.isfinite(v).all() for v in (raw,request,events,text)):
        raise ValueError('Non-finite edit input')
    if not ((lengths >= 2) & (lengths <= 104)).all():
        raise ValueError('Editing supports source lengths in [2,104]')
    valid = np.arange(104)[None] < lengths[:,None]
    tensor = lambda v: torch.as_tensor(v, device=device)
    src = tensor((raw-mean)/std * valid[...,None])
    vm, ed, text_t = tensor(valid), tensor(scope), tensor(text)
    alpha = schedule(device)
    indices = np.linspace(299,0,steps).round().astype(int).tolist()
    noises = []
    for uid in ids:
        s = int.from_bytes(hashlib.sha256(f'{seed}:{uid}'.encode()).digest()[:8], 'little') % (2**63-1)
        noises.append(torch.randn(104,40,device=device,generator=torch.Generator(device=device).manual_seed(s)))
    eps = torch.stack(noises)
    x = torch.where(ed, eps, alpha[299].sqrt()*src+(1-alpha[299]).sqrt()*eps)*vm[...,None]
    kw = dict(edit_scope=ed,edit_request=tensor(request),edit_events=tensor(events))
    for j,k in enumerate(indices):
        ts = torch.full((len(raw),),k,dtype=torch.long,device=device)
        with precision(device,kind):
            c = model(x,ts,text_t,src,vm,vm,**kw)[0]
            u = model(x,ts,torch.zeros_like(text_t),src,vm,vm,**kw)[0]
        clean = u.float()+cfg*(c.float()-u.float())
        if j+1 == len(indices):
            x, reference = clean, src
        else:
            nxt = alpha[indices[j+1]]
            residual = (x-alpha[k].sqrt()*clean)/(1-alpha[k]).sqrt().clamp_min(1e-8)
            x = nxt.sqrt()*clean+(1-nxt).sqrt()*residual
            reference = nxt.sqrt()*src+(1-nxt).sqrt()*eps
        x = torch.where(ed,x,reference)*vm[...,None]
    out = np.where(scope,x.float().cpu().numpy()*std+mean,raw)*valid[...,None]
    if not np.isfinite(out).all():
        raise ValueError('Non-finite output')
    return out.astype(np.float32)
