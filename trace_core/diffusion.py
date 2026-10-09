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

def sinusoidal_embedding(values: Tensor, dimension: int, max_period: int=10000) -> Tensor:
    half = dimension // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=values.device) / max(half, 1))
    angles = values.float().unsqueeze(-1) * frequencies
    result = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    return F.pad(result, (0, dimension - result.shape[-1]))

@dataclass(frozen=True)
class SMDContinuousDiffusionConfig:
    feature_dim: int = 38
    condition_embedding_dim: int = 1024
    max_condition_tokens: int = 9
    patch_size: int = 4
    max_patches: int = 128
    d_model: int = 448
    nhead: int = 8
    layers: int = 8
    dim_feedforward: int = 1792
    dropout: float = 0.1
    condition_dropout: float = 0.1
    reference_conditioning: bool = False
    raw_text_residual_dim: int = 0

    def validate(self) -> None:
        values = (self.feature_dim, self.condition_embedding_dim, self.max_condition_tokens, self.patch_size, self.max_patches, self.d_model, self.nhead, self.layers, self.dim_feedforward)
        if min(values) < 1:
            raise ValueError('configuration dimensions must be positive')
        if self.max_condition_tokens < 2:
            raise ValueError('condition needs one global and at least one phase slot')
        if self.d_model % self.nhead:
            raise ValueError('d_model must be divisible by nhead')
        if not 0 <= self.condition_dropout <= 1:
            raise ValueError('condition_dropout must be in [0,1]')
        if not 0 <= self.raw_text_residual_dim < self.condition_embedding_dim:
            raise ValueError('raw text residual dimension must leave a nonempty base block')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

class SMDContinuousDenoiser(nn.Module):
    """连续动作 Transformer：输入带噪动作和条件，预测归一化干净动作。"""

    def __init__(self, config: SMDContinuousDiffusionConfig):
        super().__init__()
        config.validate()
        self.config = config
        d_model = config.d_model
        self.input_projection = nn.Linear(config.feature_dim * config.patch_size, d_model)
        self.output_projection = nn.Linear(d_model, config.feature_dim * config.patch_size)
        base_condition_dim = config.condition_embedding_dim - config.raw_text_residual_dim
        self.condition_projection = nn.Sequential(nn.LayerNorm(base_condition_dim), nn.Linear(base_condition_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.condition_type = nn.Embedding(2, d_model)
        self.condition_order = nn.Embedding(config.max_condition_tokens, d_model)
        self.span_projection = nn.Sequential(nn.Linear(4, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.condition_norm = nn.LayerNorm(d_model)
        self.null_condition = nn.Parameter(torch.zeros(d_model))
        self.local_condition = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.progress_projection = nn.Sequential(nn.Linear(4, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.time_projection = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.SiLU(), nn.Linear(d_model * 4, d_model))
        self.position = nn.Embedding(config.max_patches, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=config.nhead, dim_feedforward=config.dim_feedforward, dropout=config.dropout, activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.layers, norm=nn.LayerNorm(d_model))
        self.reset_parameters()
        if config.raw_text_residual_dim:
            with torch.random.fork_rng(devices=[]):
                self.raw_text_projection = nn.Sequential(nn.LayerNorm(config.raw_text_residual_dim), nn.Linear(config.raw_text_residual_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            nn.init.zeros_(self.raw_text_projection[-1].weight)
            nn.init.zeros_(self.raw_text_projection[-1].bias)
        if config.reference_conditioning:
            with torch.random.fork_rng(devices=[]):
                self.reference_projection = nn.Linear(2 * config.feature_dim * config.patch_size, d_model)
            nn.init.zeros_(self.reference_projection.weight)
            nn.init.zeros_(self.reference_projection.bias)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.null_condition, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def patchify(self, motion: Tensor, frame_mask: Tensor) -> tuple[Tensor, Tensor, int]:
        batch, frames, channels = motion.shape
        if channels != self.config.feature_dim or frame_mask.shape != (batch, frames):
            raise ValueError('motion/frame mask dimensions differ from configuration')
        patch = self.config.patch_size
        padded = math.ceil(frames / patch) * patch
        if padded != frames:
            motion = F.pad(motion, (0, 0, 0, padded - frames))
            frame_mask = F.pad(frame_mask, (0, padded - frames), value=False)
        patches = motion.reshape(batch, padded // patch, patch * channels)
        patch_mask = frame_mask.reshape(batch, padded // patch, patch).any(-1)
        if patches.shape[1] > self.config.max_patches:
            raise ValueError('motion exceeds configured maximum patch length')
        return (patches, patch_mask, frames)

    def unpatchify(self, patches: Tensor, original_frames: int) -> Tensor:
        batch, length, _ = patches.shape
        motion = patches.reshape(batch, length * self.config.patch_size, self.config.feature_dim)
        return motion[:, :original_frames]

    def encode_condition(self, embeddings: Tensor, condition_mask: Tensor, spans: Tensor, *, force_unconditional: bool, condition_drop_mask: Tensor | None=None) -> tuple[Tensor, Tensor]:
        if embeddings.ndim != 3:
            raise ValueError('condition embeddings must have shape [B,C,D]')
        batch, conditions, dimension = embeddings.shape
        if dimension != self.config.condition_embedding_dim:
            raise ValueError('condition embedding dimension differs from config')
        if condition_mask.shape != (batch, conditions):
            raise ValueError('condition mask must have shape [B,C]')
        if spans.shape != (batch, conditions, 2):
            raise ValueError('condition spans must have shape [B,C,2]')
        if conditions > self.config.max_condition_tokens or not condition_mask[:, 0].all():
            raise ValueError('invalid condition slots/global mask')
        if self.config.raw_text_residual_dim:
            base_dim = dimension - self.config.raw_text_residual_dim
            memory = self.condition_projection(embeddings[..., :base_dim].float())
            raw = self.raw_text_projection(embeddings[:, :1, base_dim:].float())
            global_slot = torch.arange(conditions, device=memory.device).eq(0)
            memory = memory + raw * global_slot[None, :, None]
        else:
            memory = self.condition_projection(embeddings.float())
        order = torch.arange(conditions, device=memory.device)
        types = order.ne(0).long()
        normalized_spans = spans.float() / float(self.config.max_patches)
        start, end = normalized_spans.unbind(-1)
        span_features = torch.stack((start, end, end - start, (start + end) * 0.5), dim=-1)
        memory = memory + self.condition_order(order)[None]
        memory = memory + self.condition_type(types)[None]
        memory = memory + self.span_projection(span_features)
        memory = self.condition_norm(memory)
        if force_unconditional:
            dropped = torch.ones(batch, dtype=torch.bool, device=memory.device)
        elif condition_drop_mask is not None:
            if condition_drop_mask.shape != (batch,):
                raise ValueError('Condition dropout mask must have shape [B]')
            dropped = condition_drop_mask.bool()
        elif self.training and self.config.condition_dropout:
            dropped = torch.rand(batch, device=memory.device) < self.config.condition_dropout
        else:
            dropped = torch.zeros(batch, dtype=torch.bool, device=memory.device)
        memory = torch.where(dropped[:, None, None], self.null_condition[None, None], memory)
        effective = condition_mask.bool().clone()
        effective[dropped] = False
        effective[dropped, 0] = True
        return (memory, ~effective)

    def phase_progress_features(self, spans: Tensor, phase_ids: Tensor, patch_padding: Tensor) -> Tensor:
        batch, length = phase_ids.shape
        phase_slots = (phase_ids + 1).clamp(min=1)
        phase_spans = spans.gather(1, phase_slots[..., None].expand(-1, -1, 2)).float()
        start, end = phase_spans.unbind(-1)
        positions = torch.arange(length, device=phase_ids.device).float()[None]
        denominator = (end - start - 1).clamp_min(1)
        phase_progress = ((positions - start) / denominator).clamp(0, 1)
        phase_remaining = ((end - 1 - positions) / denominator).clamp(0, 1)
        motion_lengths = spans[:, 0, 1].float().clamp_min(1)
        global_progress = (positions / (motion_lengths - 1).clamp_min(1)[:, None]).clamp(0, 1)
        phase_duration = ((end - start) / motion_lengths[:, None]).clamp(0, 1)
        features = torch.stack((global_progress, phase_progress, phase_remaining, phase_duration), -1)
        return features.masked_fill(patch_padding[..., None], 0)

    def aligned_condition(self, memory: Tensor, spans: Tensor, phase_ids: Tensor, patch_padding: Tensor) -> Tensor:
        phase_slots = (phase_ids + 1).clamp(min=1)
        phase_memory = memory.gather(1, phase_slots[..., None].expand(-1, -1, self.config.d_model))
        global_memory = memory[:, :1].expand(-1, phase_ids.shape[1], -1)
        base = self.local_condition(torch.cat((global_memory, phase_memory), -1))
        progress = self.phase_progress_features(spans, phase_ids, patch_padding)
        return base + self.progress_projection(progress)

    def forward(self, noisy_motion: Tensor, timesteps: Tensor, condition_embeddings: Tensor, condition_mask: Tensor, condition_spans: Tensor, phase_ids: Tensor, frame_mask: Tensor, force_unconditional: bool=False, reference_motion: Tensor | None=None, repair_mask: Tensor | None=None, external_patch_condition: Tensor | None=None, condition_drop_mask: Tensor | None=None) -> Tensor:
        patches, patch_mask, original_frames = self.patchify(noisy_motion, frame_mask)
        batch, length = patches.shape[:2]
        if phase_ids.shape[0] != batch or phase_ids.shape[1] < length:
            raise ValueError('phase ids do not cover the padded motion patches')
        phase_ids = phase_ids[:, :length]
        patch_padding = ~patch_mask
        if (phase_ids[patch_mask] < 0).any():
            raise ValueError('valid patches must map to an SMD phase')
        memory, memory_padding = self.encode_condition(condition_embeddings, condition_mask, condition_spans, force_unconditional=force_unconditional, condition_drop_mask=condition_drop_mask)
        maximum_phase = condition_embeddings.shape[1] - 2
        if (phase_ids[patch_mask] > maximum_phase).any():
            raise ValueError('phase id has no corresponding condition slot')
        motion = self.input_projection(patches)
        if external_patch_condition is not None:
            if external_patch_condition.shape != motion.shape:
                raise ValueError('External patch condition must match motion patch shape')
            motion = motion + external_patch_condition.to(motion.dtype)
        if self.config.reference_conditioning:
            if reference_motion is None or repair_mask is None:
                raise ValueError('Reference-conditioned checkpoint requires reference_motion and repair_mask')
            if reference_motion.shape != noisy_motion.shape or repair_mask.shape != noisy_motion.shape:
                raise ValueError('Editing reference/mask must match noisy motion shape')
            valid = frame_mask[..., None]
            ref_patches, _, _ = self.patchify(reference_motion * valid, frame_mask)
            edit_patches, _, _ = self.patchify(repair_mask.to(noisy_motion.dtype) * valid, frame_mask)
            motion = motion + self.reference_projection(torch.cat((ref_patches, edit_patches), dim=-1))
        positions = self.position(torch.arange(length, device=motion.device))
        time = self.time_projection(sinusoidal_embedding(timesteps, self.config.d_model)).to(motion.dtype)
        local = self.aligned_condition(memory, condition_spans, phase_ids, patch_padding).to(motion.dtype)
        hidden = self.input_norm(motion + positions[None] + time[:, None] + local)
        hidden = self.decoder(hidden, memory.to(hidden.dtype), tgt_key_padding_mask=patch_padding, memory_key_padding_mask=memory_padding)
        output = self.unpatchify(self.output_projection(hidden), original_frames)
        return output * frame_mask.unsqueeze(-1).to(output.dtype)

@dataclass(frozen=True)
class ContinuousDiffusionConfig:
    steps: int = 1000
    lambda_velocity: float = 0.5
    lambda_acceleration: float = 0.1
    x0_clip: float = 6.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def cosine_beta_schedule(steps: int, offset: float=0.008) -> Tensor:
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos((x / steps + offset) / (1 + offset) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-08, 0.999).float()

def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    weight = mask.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum() / (weight.sum() * value.shape[-1]).clamp_min(1)

class SMDGaussianDiffusion(nn.Module):
    """余弦噪声调度与 DDIM 采样；部署入口只使用标准条件生成分支。"""

    def __init__(self, config: ContinuousDiffusionConfig):
        super().__init__()
        self.config = config
        if config.steps < 2:
            raise ValueError('diffusion steps must be at least two')
        betas = cosine_beta_schedule(config.steps)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, 0)
        self.register_buffer('alpha_bar', alpha_bar, persistent=True)
        self.register_buffer('sqrt_alpha_bar', alpha_bar.sqrt(), persistent=False)
        self.register_buffer('sqrt_one_minus_alpha_bar', (1 - alpha_bar).sqrt(), persistent=False)

    @property
    def steps(self) -> int:
        return self.config.steps

    def _extract(self, values: Tensor, timesteps: Tensor, target: Tensor) -> Tensor:
        return values.gather(0, timesteps).reshape((-1,) + (1,) * (target.ndim - 1)).to(target.dtype)

    def q_sample(self, clean: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        return self._extract(self.sqrt_alpha_bar, timesteps, clean) * clean + self._extract(self.sqrt_one_minus_alpha_bar, timesteps, clean) * noise

    def training_losses(self, model: nn.Module, clean: Tensor, timesteps: Tensor, condition_embeddings: Tensor, condition_mask: Tensor, condition_spans: Tensor, phase_ids: Tensor, frame_mask: Tensor, noise: Tensor | None=None) -> dict[str, Tensor]:
        noise = torch.randn_like(clean) if noise is None else noise
        noisy = self.q_sample(clean, timesteps, noise)
        noisy = noisy * frame_mask.unsqueeze(-1).to(noisy.dtype)
        prediction = model(noisy, timesteps, condition_embeddings, condition_mask, condition_spans, phase_ids, frame_mask)
        reconstruction = masked_mean((prediction - clean) ** 2, frame_mask)
        pair_mask = frame_mask[:, 1:] & frame_mask[:, :-1]
        velocity = masked_mean((prediction[:, 1:] - prediction[:, :-1] - (clean[:, 1:] - clean[:, :-1])) ** 2, pair_mask)
        triple_mask = pair_mask[:, 1:] & pair_mask[:, :-1]
        pred_acc = prediction[:, 2:] - 2 * prediction[:, 1:-1] + prediction[:, :-2]
        clean_acc = clean[:, 2:] - 2 * clean[:, 1:-1] + clean[:, :-2]
        acceleration = masked_mean((pred_acc - clean_acc) ** 2, triple_mask)
        total = reconstruction + self.config.lambda_velocity * velocity + self.config.lambda_acceleration * acceleration
        return {'loss': total, 'x0_mse': reconstruction.detach(), 'velocity_mse': velocity.detach(), 'acceleration_mse': acceleration.detach(), 'prediction': prediction}

    def project_counterfactual_clean(self, clean: Tensor, output_mask: Tensor | None=None) -> Tensor:
        return clean

    @torch.inference_mode()
    def ddim_sample(self, model: nn.Module, shape: tuple[int, int, int], condition_embeddings: Tensor, condition_mask: Tensor, condition_spans: Tensor, phase_ids: Tensor, frame_mask: Tensor, *, sampling_steps: int=50, guidance_scale: float=1.5, generator: torch.Generator | None=None, initial_noise: Tensor | None=None, counterfactual_condition_embeddings: Tensor | None=None, counterfactual_condition_mask: Tensor | None=None, counterfactual_condition_spans: Tensor | None=None, counterfactual_phase_ids: Tensor | None=None, counterfactual_output_mask: Tensor | None=None, counterfactual_guidance_scale: float=0.0, counterfactual_trust_ratio: float=1.0, counterfactual_schedule: str='constant', counterfactual_reference: str='plan', counterfactual_diagnostics: dict[str, Any] | None=None) -> Tensor:
        """执行逐步去噪，使用条件与无条件预测构成 CFG，保持原始计算顺序。"""
        if sampling_steps < 1:
            raise ValueError('sampling_steps must be positive')
        if guidance_scale < 0:
            raise ValueError('guidance_scale must be non-negative')
        if counterfactual_guidance_scale < 0:
            raise ValueError('counterfactual_guidance_scale must be non-negative')
        if counterfactual_trust_ratio <= 0:
            raise ValueError('counterfactual_trust_ratio must be positive')
        if counterfactual_schedule not in {'constant', 'early_cosine'}:
            raise ValueError("counterfactual_schedule must be 'constant' or 'early_cosine'")
        if counterfactual_reference not in {'plan', 'unconditional'}:
            raise ValueError("counterfactual_reference must be 'plan' or 'unconditional'")
        counterfactual_values = (counterfactual_condition_embeddings, counterfactual_condition_mask, counterfactual_condition_spans, counterfactual_phase_ids, counterfactual_output_mask)
        counterfactual_active = counterfactual_guidance_scale > 0
        if counterfactual_active and any((value is None for value in counterfactual_values)):
            raise ValueError('active counterfactual guidance requires negative condition, spans, phase ids, and output mask')
        if not counterfactual_active and any((value is not None for value in counterfactual_values)):
            raise ValueError('counterfactual tensors were supplied with zero guidance scale')
        if counterfactual_active:
            assert counterfactual_condition_embeddings is not None
            assert counterfactual_condition_mask is not None
            assert counterfactual_condition_spans is not None
            assert counterfactual_phase_ids is not None
            assert counterfactual_output_mask is not None
            expected_shapes = ((counterfactual_condition_embeddings, condition_embeddings.shape, 'counterfactual_condition_embeddings'), (counterfactual_condition_mask, condition_mask.shape, 'counterfactual_condition_mask'), (counterfactual_condition_spans, condition_spans.shape, 'counterfactual_condition_spans'), (counterfactual_phase_ids, phase_ids.shape, 'counterfactual_phase_ids'), (counterfactual_output_mask, shape, 'counterfactual_output_mask'))
            for value, expected, name in expected_shapes:
                if tuple(value.shape) != tuple(expected):
                    raise ValueError(f'{name} shape {tuple(value.shape)} differs from {tuple(expected)}')
            if not torch.isfinite(counterfactual_output_mask).all():
                raise ValueError('counterfactual_output_mask must be finite')
            if (counterfactual_output_mask < 0).any() or (counterfactual_output_mask > 1).any():
                raise ValueError('counterfactual_output_mask weights must be in [0,1]')
        device = condition_embeddings.device
        if initial_noise is None:
            sample = torch.randn(shape, device=device, generator=generator)
        else:
            if tuple(initial_noise.shape) != tuple(shape):
                raise ValueError(f'initial_noise shape {tuple(initial_noise.shape)} differs from requested sample shape {tuple(shape)}')
            sample = initial_noise.to(device=device).clone()
        sample = sample * frame_mask.unsqueeze(-1).to(sample.dtype)
        schedule = torch.linspace(self.steps - 1, 0, sampling_steps, device=device).round().long().unique_consecutive()
        trust_scales: list[float] = []
        saturation_fractions: list[float] = []
        for index, timestep in enumerate(schedule):
            times = torch.full((shape[0],), int(timestep), device=device, dtype=torch.long)
            conditional = model(sample, times, condition_embeddings, condition_mask, condition_spans, phase_ids, frame_mask)
            if guidance_scale != 1 or counterfactual_active:
                unconditional = model(sample, times, condition_embeddings, condition_mask, condition_spans, phase_ids, frame_mask, force_unconditional=True)
                if counterfactual_active:
                    clean = unconditional.float() + guidance_scale * (conditional.float() - unconditional.float())
                else:
                    clean = unconditional + guidance_scale * (conditional - unconditional)
            else:
                clean = conditional
            if counterfactual_active:
                assert counterfactual_condition_embeddings is not None
                assert counterfactual_condition_mask is not None
                assert counterfactual_condition_spans is not None
                assert counterfactual_phase_ids is not None
                assert counterfactual_output_mask is not None
                if counterfactual_reference == 'plan':
                    negative = model(sample, times, counterfactual_condition_embeddings, counterfactual_condition_mask, counterfactual_condition_spans, counterfactual_phase_ids, frame_mask)
                else:
                    negative = unconditional
                local_mask = counterfactual_output_mask.to(device=device, dtype=torch.float32) * frame_mask.unsqueeze(-1).to(torch.float32)
                local_count = (local_mask > 0).sum(dim=(1, 2)).clamp_min(1)
                contrast = local_mask * (conditional.float() - negative.float())
                cfg_residual = local_mask * (conditional.float() - unconditional.float())
                contrast_rms = torch.sqrt(contrast.square().sum(dim=(1, 2)) / local_count)
                cfg_rms = torch.sqrt(cfg_residual.square().sum(dim=(1, 2)) / local_count)
                trust = torch.minimum(torch.ones_like(contrast_rms), counterfactual_trust_ratio * cfg_rms / contrast_rms.clamp_min(1e-08))
                if counterfactual_schedule == 'early_cosine':
                    progress = float(timestep) / max(self.steps - 1, 1)
                    schedule_weight = math.sin(0.5 * math.pi * progress) ** 2
                else:
                    schedule_weight = 1.0
                clean = clean.float() + counterfactual_guidance_scale * schedule_weight * trust[:, None, None] * contrast
                trust_scales.extend(trust.detach().float().cpu().tolist())
                saturation_fractions.append(float((clean.abs() > self.config.x0_clip).float().mean().item()))
            clean = clean.clamp(-self.config.x0_clip, self.config.x0_clip)
            if counterfactual_active:
                clean = self.project_counterfactual_clean(clean, local_mask).clamp(-self.config.x0_clip, self.config.x0_clip)
            if index == len(schedule) - 1:
                sample = clean
                continue
            next_t = schedule[index + 1]
            alpha = self.alpha_bar[timestep].to(sample.dtype)
            next_alpha = self.alpha_bar[next_t].to(sample.dtype)
            epsilon = (sample - alpha.sqrt() * clean) / (1 - alpha).sqrt().clamp_min(1e-08)
            sample = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * epsilon
            sample = sample * frame_mask.unsqueeze(-1).to(sample.dtype)
        if counterfactual_diagnostics is not None:
            counterfactual_diagnostics.clear()
            counterfactual_diagnostics.update({'active': counterfactual_active, 'guidance_scale': float(counterfactual_guidance_scale), 'trust_ratio': float(counterfactual_trust_ratio), 'schedule': counterfactual_schedule, 'reference': counterfactual_reference, 'mean_trust_scale': float(np.mean(trust_scales)) if trust_scales else None, 'minimum_trust_scale': float(np.min(trust_scales)) if trust_scales else None, 'mean_pre_clamp_saturation_fraction': float(np.mean(saturation_fractions)) if saturation_fractions else 0.0, 'maximum_pre_clamp_saturation_fraction': float(np.max(saturation_fractions)) if saturation_fractions else 0.0, 'sampling_steps': len(schedule)})
        return sample

    @torch.inference_mode()
    def ddim_inpaint_sample(self, model: nn.Module, reference_clean: Tensor, repair_mask: Tensor, condition_embeddings: Tensor, condition_mask: Tensor, condition_spans: Tensor, phase_ids: Tensor, frame_mask: Tensor, *, sampling_steps: int=50, guidance_scale: float=1.5, generator: torch.Generator | None=None, start_fraction: float=1.0) -> Tensor:
        if reference_clean.ndim != 3:
            raise ValueError('reference_clean must have [B,T,D] shape')
        if repair_mask.shape != reference_clean.shape:
            raise ValueError('repair_mask shape must match reference_clean')
        if frame_mask.shape != reference_clean.shape[:2]:
            raise ValueError('frame_mask shape differs from reference motion')
        if sampling_steps < 1:
            raise ValueError('sampling_steps must be positive')
        if not 0 < start_fraction <= 1:
            raise ValueError('start_fraction must be in (0,1]')
        device = condition_embeddings.device
        reference_clean = reference_clean.to(device)
        repair_mask = repair_mask.to(device=device, dtype=torch.bool)
        valid = frame_mask.to(device=device, dtype=torch.bool).unsqueeze(-1)
        repair_mask = repair_mask & valid
        known_mask = valid & ~repair_mask
        shape = tuple(reference_clean.shape)
        reference_noise = torch.randn(shape, device=device, dtype=reference_clean.dtype, generator=generator)
        schedule = torch.linspace(round((self.steps - 1) * start_fraction), 0, sampling_steps, device=device).round().long().unique_consecutive()
        initial_times = torch.full((shape[0],), int(schedule[0]), device=device, dtype=torch.long)
        sample = self.q_sample(reference_clean, initial_times, reference_noise)
        sample = sample * valid.to(sample.dtype)
        if getattr(getattr(model, 'config', None), 'reference_conditioning', False):
            model = partial(model, reference_motion=reference_clean, repair_mask=repair_mask)
        for index, timestep in enumerate(schedule):
            times = torch.full((shape[0],), int(timestep), device=device, dtype=torch.long)
            conditional = model(sample, times, condition_embeddings, condition_mask, condition_spans, phase_ids, frame_mask)
            if guidance_scale != 1:
                unconditional = model(sample, times, condition_embeddings, condition_mask, condition_spans, phase_ids, frame_mask, force_unconditional=True)
                clean = unconditional + guidance_scale * (conditional - unconditional)
            else:
                clean = conditional
            clean = clean.clamp(-self.config.x0_clip, self.config.x0_clip)
            if index == len(schedule) - 1:
                sample = torch.where(known_mask, reference_clean, clean)
                sample = sample * valid.to(sample.dtype)
                continue
            next_t = schedule[index + 1]
            alpha = self.alpha_bar[timestep].to(sample.dtype)
            next_alpha = self.alpha_bar[next_t].to(sample.dtype)
            epsilon = (sample - alpha.sqrt() * clean) / (1 - alpha).sqrt().clamp_min(1e-08)
            sample = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * epsilon
            next_times = torch.full((shape[0],), int(next_t), device=device, dtype=torch.long)
            known_noisy = self.q_sample(reference_clean, next_times, reference_noise)
            sample = torch.where(known_mask, known_noisy, sample)
            sample = sample * valid.to(sample.dtype)
        return torch.where(known_mask, reference_clean, sample)
