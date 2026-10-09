"""真实首尾片段条件扩散；独立扩展，不修改已发布生成器。"""
import torch
from torch import nn
from .events import EventDenoiser


class BoundaryEventDenoiser(EventDenoiser):
    def __init__(self,config):
        super().__init__(config)
        self.boundary_projection=nn.Linear(config.patch_size*config.feature_dim*2,config.d_model)
        nn.init.zeros_(self.boundary_projection.weight);nn.init.zeros_(self.boundary_projection.bias)

    def forward(self,x,t,embeddings,mask,spans,phase_ids,frame_mask,force_unconditional=False,
                boundary_reference=None,boundary_mask=None):
        d=self.config.condition_embedding_dim;batch=len(x)
        raw=embeddings[...,:d];patches=(x.shape[1]+self.config.patch_size-1)//self.config.patch_size
        probability=embeddings[:,0,d:d+2048].reshape(batch,128,16)[:,:patches]
        dropped=torch.ones(batch,dtype=torch.bool,device=x.device) if force_unconditional else (
            torch.rand(batch,device=x.device)<self.config.condition_dropout if self.training else
            torch.zeros(batch,dtype=torch.bool,device=x.device))
        _,valid,_=self.backbone.patchify(x,frame_mask);ids=torch.zeros_like(phase_ids)
        progress=self.backbone.phase_progress_features(spans,ids[:,:patches],~valid)
        caption=raw[:,:1].expand(-1,patches,-1)
        residual=self.raw_projection(torch.cat([caption,caption,progress],-1))+self.contact_projection(torch.cat([probability,progress],-1))
        residual=residual*(~dropped[:,None,None])*valid[:,:,None]
        residual=residual*(t.float()/((self.steps-1)*self.low_noise_fraction)).clamp(0,1)[:,None,None].to(residual.dtype)
        if boundary_reference is None:boundary_reference=torch.zeros_like(x)
        if boundary_mask is None:boundary_mask=torch.zeros_like(x,dtype=torch.bool)
        if boundary_reference.shape!=x.shape or boundary_mask.shape!=x.shape:raise ValueError('首尾条件与动作维度不一致')
        known=boundary_mask.bool()&frame_mask[...,None]
        # 未知区域先清零，绝不把 GT 中间动作泄漏给条件分支。
        reference=torch.where(known,boundary_reference,torch.zeros_like(boundary_reference))
        rp,_,_=self.backbone.patchify(reference,frame_mask)
        mp,_,_=self.backbone.patchify(known.to(reference.dtype),frame_mask)
        bc=self.boundary_projection(torch.cat([rp,mp],-1))*(mp.any(-1)&valid)[...,None]
        # CFG 仅丢弃语义条件；首尾姿态条件始终保留，低噪声时也不衰减。
        return self.backbone(x,t,raw,mask,spans,ids,frame_mask,force_unconditional=force_unconditional,
            condition_drop_mask=dropped,external_patch_condition=residual+bc)


def sample_with_boundaries(model,diff,batch,reference,known,steps=50,cfg=2.5,generator=None, precision="bf16"):
    """只读取已知区域；每步使用同一参考噪声，最后精确恢复已知片段。"""
    valid=batch['frame_mask'][...,None];known=known.bool()&valid
    reference=torch.where(known,reference,torch.zeros_like(reference))
    noise=torch.randn(reference.shape,device=reference.device,dtype=reference.dtype,generator=generator)
    schedule=torch.linspace(diff.steps-1,0,steps,device=reference.device).round().long().unique_consecutive()
    times=torch.full((len(reference),),int(schedule[0]),device=reference.device,dtype=torch.long)
    x=torch.where(known,diff.q_sample(reference,times,noise),noise)*valid
    args=[batch[k] for k in ('condition_embeddings','condition_mask','condition_spans','phase_ids','frame_mask')]
    for i,t in enumerate(schedule):
        ts=torch.full((len(reference),),int(t),device=x.device,dtype=torch.long)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=x.is_cuda and precision=='bf16'):
            conditional=model(x,ts,*args,boundary_reference=reference,boundary_mask=known)
            unconditional=model(x,ts,*args,boundary_reference=reference,boundary_mask=known,force_unconditional=True)
        clean=(unconditional.float()+cfg*(conditional.float()-unconditional.float())).clamp(-diff.config.x0_clip,diff.config.x0_clip)
        clean=torch.where(known,reference,clean)
        if i+1==len(schedule):x=clean
        else:
            alpha=diff.alpha_bar[t];nxt=diff.alpha_bar[schedule[i+1]]
            eps=(x-alpha.sqrt()*clean)/(1-alpha).sqrt().clamp_min(1e-8)
            x=nxt.sqrt()*clean+(1-nxt).sqrt()*eps
            x=torch.where(known,nxt.sqrt()*reference+(1-nxt).sqrt()*noise,x)
        x=x*valid
    return torch.where(known,reference,x)

