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
from .diffusion import SMDContinuousDenoiser

class ContactConditionDenoiser(nn.Module):
    """事件生成器的公共骨架；参数名必须与训练权重一致。"""

    def __init__(self, config, low_noise_fraction=0.25, diffusion_steps=1000):
        super().__init__()
        self.config = config
        self.low_noise_fraction = low_noise_fraction
        self.steps = diffusion_steps
        self.backbone = SMDContinuousDenoiser(config)
        with torch.random.fork_rng(devices=[]):
            self.raw_projection = nn.Sequential(nn.LayerNorm(config.condition_embedding_dim * 2 + 4), nn.Linear(config.condition_embedding_dim * 2 + 4, config.d_model), nn.SiLU(), nn.Linear(config.d_model, config.d_model))
            self.contact_projection = nn.Sequential(nn.Linear(6, 128), nn.SiLU(), nn.Linear(128, config.d_model))
        for module in (self.raw_projection, self.contact_projection):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)

    def forward(self, x, t, embeddings, mask, spans, phase_ids, frame_mask, force_unconditional=False):
        d = self.config.condition_embedding_dim
        batch = len(x)
        assert d == 2560 and embeddings.shape[-1] == 2 * d
        raw = embeddings[..., :d]
        patches = (x.shape[1] + self.config.patch_size - 1) // self.config.patch_size
        probability = embeddings[:, 0, d:d + 256].reshape(batch, 128, 2)[:, :patches]
        dropped = torch.ones(batch, dtype=torch.bool, device=x.device) if force_unconditional else torch.rand(batch, device=x.device) < self.config.condition_dropout if self.training else torch.zeros(batch, dtype=torch.bool, device=x.device)
        _, valid, _ = self.backbone.patchify(x, frame_mask)
        ids = torch.zeros_like(phase_ids)
        progress = self.backbone.phase_progress_features(spans, ids[:, :patches], ~valid)
        caption = raw[:, :1].expand(-1, patches, -1)
        residual = self.raw_projection(torch.cat([caption, caption, progress], -1))
        residual = residual + self.contact_projection(torch.cat([probability, progress], -1))
        residual = residual * ~dropped[:, None, None] * valid[:, :, None]
        if self.low_noise_fraction:
            scale = (t.float() / ((self.steps - 1) * self.low_noise_fraction)).clamp(0, 1)
            residual = residual * scale[:, None, None].to(residual.dtype)
        return self.backbone(x, t, raw, mask, spans, ids, frame_mask, force_unconditional=force_unconditional, condition_drop_mask=dropped, external_patch_condition=residual)
