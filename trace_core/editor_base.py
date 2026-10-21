from __future__ import annotations
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any
import math
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .diffusion import SMDContinuousDenoiser, SMDContinuousDiffusionConfig


GROUPS = ((2, 5), (5, 8), (8, 12), (12, 16))


class SourcePlanEditor(nn.Module):
    """Encode unwarped source tokens and predict a text-conditioned target Plan."""
    def __init__(self, plan=True, scope_condition=False, request_condition=False):
        super().__init__()
        self.use_plan = plan
        cfg = SMDContinuousDiffusionConfig(feature_dim=40, condition_embedding_dim=2560,
             max_condition_tokens=2, max_patches=26, d_model=448, nhead=8, layers=8,
             dim_feedforward=1792, condition_dropout=0.)
        self.backbone = SMDContinuousDenoiser(cfg)
        self.source_projection = nn.Linear(160, 448)
        self.source_position = nn.Embedding(26, 448)
        self.query_position = nn.Embedding(26, 448)
        self.text_query = nn.Linear(2560, 448)
        self.source_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            448, 8, 1792, dropout=.1, activation='gelu', batch_first=True), 2,
            enable_nested_tensor=False)
        self.source_attention = nn.MultiheadAttention(448, 8, dropout=.1, batch_first=True)
        self.event_head = nn.Sequential(nn.LayerNorm(448), nn.Linear(448, 16))
        self.event_projection = nn.Linear(16, 448)
        nn.init.zeros_(self.event_projection.weight)
        nn.init.zeros_(self.event_projection.bias)
        self.length_head = nn.Sequential(nn.LayerNorm(448), nn.Linear(448, 1))
        self.use_scope = scope_condition
        if scope_condition:
            self.scope_projection = nn.Linear(160, 448)
            nn.init.zeros_(self.scope_projection.weight)
            nn.init.zeros_(self.scope_projection.bias)
        self.use_request = request_condition
        if request_condition:
            if not scope_condition:
                raise ValueError('Relative requests require explicit scope conditioning')
            self.request_projection = nn.Linear(8, 448)
            nn.init.zeros_(self.request_projection.weight)
            nn.init.zeros_(self.request_projection.bias)

    def forward(self, x, timesteps, text, source, source_mask, target_mask, edit_scope=None, edit_request=None):
        b, t, _ = x.shape
        patches = t // 4
        src, sm, _ = self.backbone.patchify(source, source_mask)
        src = self.source_projection(src) + self.source_position(torch.arange(src.shape[1], device=x.device))[None]
        src = self.source_encoder(src, src_key_padding_mask=~sm)
        query = self.query_position(torch.arange(patches, device=x.device))[None] + self.text_query(text)[:, None]
        context = self.source_attention(query, src, src, key_padding_mask=~sm, need_weights=False)[0]
        logits = self.event_head(context + query).float()
        prob = torch.cat([logits[..., :2].sigmoid()] + [logits[..., a:z].softmax(-1) for a, z in GROUPS], -1)
        external = context + (self.event_projection(prob) if self.use_plan else 0)
        if self.use_scope:
            if edit_scope is None or edit_scope.shape != x.shape:
                raise ValueError('Scope-conditioned editor requires a matching body-time mask')
            scoped = edit_scope.float() * target_mask[..., None]
            external = external + self.scope_projection(scoped.reshape(b, patches, 160))
        if self.use_request:
            if edit_request is None or edit_request.shape != (b, t, 2):
                raise ValueError('Relative-request editor requires two wrist-offset channels')
            requested = edit_request.float() / .05 * target_mask[..., None]
            external = external + self.request_projection(requested.reshape(b, patches, 8))
        condition = text[:, None].expand(-1, 2, -1)
        mask = torch.ones(b, 2, dtype=torch.bool, device=x.device)
        spans = torch.zeros(b, 2, 2, dtype=torch.long, device=x.device)
        spans[:, :, 1] = ((target_mask.sum(1)+3)//4)[:, None]
        ids = torch.zeros(b, patches, dtype=torch.long, device=x.device)
        output = self.backbone(x, timesteps, condition, mask, spans, ids, target_mask,
                               external_patch_condition=external)
        pooled = (src * sm[..., None]).sum(1) / sm.sum(1, keepdim=True)
        length_ratio = self.length_head(pooled + self.text_query(text)).squeeze(-1)
        return output, logits, length_ratio


def schedule(device):
    s = torch.arange(301, dtype=torch.float64)
    a = torch.cos((s/300+.008)/1.008*math.pi/2)**2
    a = a/a[0]
    beta = (1-a[1:]/a[:-1]).clamp(0, .999)
    return torch.cumprod(1-beta, 0).float().to(device)
