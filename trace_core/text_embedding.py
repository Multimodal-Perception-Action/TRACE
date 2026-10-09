# 本模块由当前推理实现抽取；仅整理依赖和中文注释，计算逻辑保持一致。
from __future__ import annotations
from typing import Any, Sequence
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
DEFAULT_INSTRUCTION = 'Encode one part of a structured humanoid motion plan. Preserve body side, travel and turn direction, phase order and span, repetition, duration, support state, root change, and relative limb transitions.'

def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """提取最后一个有效 token 的隐藏状态。"""
    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return hidden[:, -1]
    positions = attention_mask.sum(1) - 1
    return hidden[torch.arange(len(hidden), device=hidden.device), positions]

def encode_texts(texts: Sequence[str], tokenizer: Any, model: Any, device: torch.device, dtype: torch.dtype, max_length: int, instruction: str) -> tuple[np.ndarray, int, int]:
    """采用训练时的指令模板、最后有效 token 池化和 L2 归一化。"""
    encoded = tokenizer([f'Instruct: {instruction}\nQuery: {text}' for text in texts], padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
    lengths = encoded['attention_mask'].sum(1)
    context = torch.autocast(device.type, dtype=dtype) if device.type == 'cuda' and dtype != torch.float32 else nullcontext()
    with torch.inference_mode(), context:
        output = model(**encoded, return_dict=True)
        pooled = F.normalize(last_token_pool(output.last_hidden_state, encoded['attention_mask']).float(), p=2, dim=1)
    return (pooled.cpu().numpy().astype(np.float16), int(lengths.max()), int(lengths.eq(max_length).sum()))
