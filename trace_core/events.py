# 本模块由当前推理实现抽取；仅整理依赖和中文注释，计算逻辑保持一致。
from __future__ import annotations
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any
import math
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .contact import ContactConditionDenoiser

def pack_events(row, probability):
    """将原始文本向量和补齐到 128 个片段的事件概率封装为模型条件。"""
    n = int(row['num_frames'])
    patches = (n + 3) // 4
    p = np.asarray(probability, dtype=np.float32)
    assert p.shape == (128, 16) and np.isfinite(p).all() and ((p >= 0) & (p <= 1.00001)).all()
    raw = row['condition_embeddings'][0]
    assert raw.shape == (2560,)
    e = torch.zeros(9, 5120)
    e[:, :2560] = raw
    e[0, 2560:4608] = torch.from_numpy(p.copy().reshape(-1))
    spans = torch.zeros(9, 2, dtype=torch.long)
    spans[:2, 1] = patches
    mask = torch.zeros(9, dtype=torch.bool)
    mask[:2] = True
    result = dict(row)
    result.update(condition_embeddings=e, condition_spans=spans, condition_mask=mask, phase_ids=torch.zeros(patches, dtype=torch.long))
    return result

class EventDenoiser(ContactConditionDenoiser):
    """事件条件生成器：文本残差与时间对齐的事件残差共同注入动作 patch。"""

    def __init__(self, config, low_noise_fraction=0.25, diffusion_steps=1000):
        super().__init__(config, low_noise_fraction, diffusion_steps)
        with torch.random.fork_rng(devices=[]):
            self.contact_projection = nn.Sequential(nn.Linear(20, 128), nn.SiLU(), nn.Linear(128, config.d_model))
            nn.init.zeros_(self.contact_projection[-1].weight)
            nn.init.zeros_(self.contact_projection[-1].bias)

    def forward(self, x, t, embeddings, mask, spans, phase_ids, frame_mask, force_unconditional=False):
        d = self.config.condition_embedding_dim
        batch = len(x)
        raw = embeddings[..., :d]
        patches = (x.shape[1] + self.config.patch_size - 1) // self.config.patch_size
        probability = embeddings[:, 0, d:d + 2048].reshape(batch, 128, 16)[:, :patches]
        dropped = torch.ones(batch, dtype=torch.bool, device=x.device) if force_unconditional else torch.rand(batch, device=x.device) < self.config.condition_dropout if self.training else torch.zeros(batch, dtype=torch.bool, device=x.device)
        _, valid, _ = self.backbone.patchify(x, frame_mask)
        ids = torch.zeros_like(phase_ids)
        progress = self.backbone.phase_progress_features(spans, ids[:, :patches], ~valid)
        caption = raw[:, :1].expand(-1, patches, -1)
        residual = self.raw_projection(torch.cat([caption, caption, progress], -1)) + self.contact_projection(torch.cat([probability, progress], -1))
        residual = residual * ~dropped[:, None, None] * valid[:, :, None]
        residual = residual * (t.float() / ((self.steps - 1) * self.low_noise_fraction)).clamp(0, 1)[:, None, None].to(residual.dtype)
        return self.backbone(x, t, raw, mask, spans, ids, frame_mask, force_unconditional=force_unconditional, condition_drop_mask=dropped, external_patch_condition=residual)
