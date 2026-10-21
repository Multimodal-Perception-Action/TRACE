"""Shared 16-channel event vocabulary for source-relative G1 editing."""
import numpy as np
import torch
from torch import nn
from .editor_base import SourcePlanEditor
from .editor_events import PalmFK, label_extra, SCHEMA
from .editor_geometry import G1SoleForwardKinematics, infer_clean_foot_contacts


class EventRevisionEditor(SourcePlanEditor):
    def __init__(self):
        super().__init__(plan=False, scope_condition=True, request_condition=True)
        # Same input dimensions and architecture as the generation event adapter.
        # Zero output preserves the old editor before any new optimization.
        self.revision_projection = nn.Sequential(nn.Linear(20, 128), nn.SiLU(), nn.Linear(128, 448))
        nn.init.zeros_(self.revision_projection[-1].weight)
        nn.init.zeros_(self.revision_projection[-1].bias)

    def forward(self, x, timesteps, text, source, source_mask, target_mask,
                edit_scope=None, edit_request=None, edit_events=None):
        b, t, _ = x.shape
        patches = t // 4
        src, sm, _ = self.backbone.patchify(source, source_mask)
        src = self.source_projection(src) + self.source_position(torch.arange(src.shape[1], device=x.device))[None]
        src = self.source_encoder(src, src_key_padding_mask=~sm)
        query = self.query_position(torch.arange(patches, device=x.device))[None] + self.text_query(text)[:, None]
        context = self.source_attention(query, src, src, key_padding_mask=~sm, need_weights=False)[0]
        logits = self.event_head(context + query).float()
        assert edit_scope is not None and edit_scope.shape == x.shape
        assert edit_request is not None and edit_request.shape == (b, t, 2)
        external = context + self.scope_projection((edit_scope.float() * target_mask[..., None]).reshape(b, patches, 160))
        external = external + self.request_projection((edit_request.float() / .05 * target_mask[..., None]).reshape(b, patches, 8))
        if edit_events is None:
            edit_events = x.new_zeros(b, patches, 16)
        assert edit_events.shape == (b, patches, 16)
        count = (target_mask.sum(1) + 3) // 4
        p = (torch.arange(patches, device=x.device)[None] / (count - 1).clamp_min(1)[:, None]).clamp(0, 1)
        progress = torch.stack((p, p, 1-p, torch.ones_like(p)), -1)
        valid = torch.arange(patches, device=x.device)[None] < count[:, None]
        events = self.revision_projection(torch.cat((edit_events, progress), -1))
        # Normalize by this editor's 300-step schedule, not the generator's 1000.
        scale = (timesteps.float() / (299 * .25)).clamp(0, 1)[:, None, None]
        external = external + events * scale * valid[..., None]
        condition = text[:, None].expand(-1, 2, -1)
        mask = torch.ones(b, 2, dtype=torch.bool, device=x.device)
        spans = torch.zeros(b, 2, 2, dtype=torch.long, device=x.device)
        spans[:, :, 1] = count[:, None]
        ids = torch.zeros(b, patches, dtype=torch.long, device=x.device)
        output = self.backbone(x, timesteps, condition, mask, spans, ids, target_mask, external_patch_condition=external)
        pooled = (src * sm[..., None]).sum(1) / sm.sum(1, keepdim=True)
        ratio = self.length_head(pooled + self.text_query(text)).squeeze(-1)
        return output, logits, ratio


class SourceEventExtractor:
    """Use the exact generation palm/sole definitions on a known source motion."""
    def __init__(self, device='cuda', xml=None):
        self.device = device
        self.palm = PalmFK(xml=xml).to(device).eval()
        self.feet = G1SoleForwardKinematics(100.).to(device).eval()

    @torch.inference_mode()
    def __call__(self, source, lengths):
        x = torch.as_tensor(source, device=self.device, dtype=torch.float32)
        valid = torch.arange(x.shape[1], device=x.device)[None] < torch.as_tensor(lengths, device=x.device)[:, None]
        palms = self.palm(x, valid)[..., 2].cpu().numpy()
        contact, ground, _ = infer_clean_foot_contacts(self.feet(x, valid), valid,
            fps=20., height_threshold_m=.035, speed_threshold_mps=.20, ground_quantile=.05)
        contact, ground = contact.cpu().numpy(), ground.cpu().numpy()
        events = np.zeros((len(x), 26, 16), np.float32)
        confidence = np.zeros_like(events)
        for i, n in enumerate(lengths):
            target, w = label_extra(source[i, :n, 2] / 100., palms[i, :n], ground[i])
            events[i, :, 2:] = target[:26]
            confidence[i, :, 2:] = w[:26]
            for a, z in ((2, 5), (5, 8), (8, 12), (12, 16)):
                unknown = events[i, :, a:z].sum(-1) == 0
                events[i, unknown, a:z] = 1 / (z-a)
            for k, start in enumerate(range(0, int(n), 4)):
                events[i, k, :2] = contact[i, start:min(start+4, n)].mean(0)
                confidence[i, k, :2] = 1
            events[i, (int(n)+3)//4:] = 0
        return events, confidence, palms, ground


def revise_events(source, lengths, source_events, source_confidence, palms, ground, request):
    """Update only requested hand groups using source geometry + explicit offsets.

    The continuous request remains separate when the categorical event is unchanged.
    Palm-category targets and wrist-position goals are deliberately distinguished.
    """
    assert request.shape == (len(source), 104, 2)
    events, weight = source_events.copy(), source_confidence.copy()
    for i, n in enumerate(lengths):
        target, w = label_extra(source[i, :n, 2]/100., palms[i, :n] + request[i, :n], ground[i])
        for side, (lo, hi) in enumerate(((8, 12), (12, 16))):
            if not np.any(request[i, :, side]):
                continue
            # Replace whole categorical groups, never individual one-hot coordinates.
            events[i, :, lo:hi] = target[:26, lo-2:hi-2]
            weight[i, :, lo:hi] = w[:26, lo-2:hi-2]
            unknown = events[i, :, lo:hi].sum(-1) == 0
            events[i, unknown, lo:hi] = .25
        events[i, (int(n)+3)//4:] = 0
    return events, weight


def soft_hand_events(palms_z):
    """Smooth four-bin classification, for differentiable event consistency only."""
    x = palms_z.transpose(1, 2).reshape(-1, 1, palms_z.shape[1])
    x = torch.nn.functional.avg_pool1d(torch.nn.functional.pad(x, (2, 2), mode='replicate'), 5, stride=1)
    x = x.reshape(palms_z.shape[0], 2, -1).transpose(1, 2)
    c = torch.sigmoid((x[..., None] - x.new_tensor([0., .25, .45])) / .025)
    p = torch.stack((1-c[..., 0], c[..., 0]-c[..., 1], c[..., 1]-c[..., 2], c[..., 2]), -1)
    return p.reshape(len(p), 26, 4, 2, 4).mean(2).reshape(len(p), 26, 8).clamp_min(1e-6)
