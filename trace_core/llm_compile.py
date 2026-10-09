# 本模块由当前推理实现抽取；仅整理依赖和中文注释，计算逻辑保持一致。
import json
import numpy as np
from .llm_plan import parse, compile_plan
GROUPS = ((2, 5), (5, 8), (8, 12), (12, 16))

def canonical_parse(response):
    """只合并相邻相同事件，不修改预测时长或类别。"""
    try:
        return (parse(response), False)
    except (ValueError, TypeError, KeyError) as original:
        obj = json.loads(response)
        if not isinstance(obj, dict) or set(obj) != {'frames', 'events'}:
            raise original
        if not isinstance(obj['events'], list):
            raise original
        merged, previous_end = ([], 0)
        for row in obj['events']:
            if not isinstance(row, list) or len(row) != 7 or any((type(x) is not int for x in row)):
                raise original
            if row[0] <= previous_end:
                raise original
            previous_end = row[0]
            if merged and row[1:] == merged[-1][1:]:
                merged[-1][0] = row[0]
            else:
                merged.append(row.copy())
        result = parse(json.dumps({'frames': obj['frames'], 'events': merged}))
        return (result, True)

def neutral(n):
    """未知支撑取 0.5，未知互斥类别取均匀分布。"""
    result = np.zeros((128, 16), np.float32)
    patches = (n + 3) // 4
    result[:patches, :2] = 0.5
    for a, b in GROUPS:
        result[:patches, a:b] = 1 / (b - a)
    return result

def compile_response(response, frames=None):
    """编译响应并返回失败标志；工程主入口会拒绝失败结果。"""
    error, canonicalized, obj = (None, False, None)
    try:
        obj, canonicalized = canonical_parse(response)
        predicted = obj['frames']
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        error = repr(exc)
        predicted = 128
    n = predicted if frames is None else int(frames)
    if not 1 <= n <= 512:
        raise ValueError('Requested length is outside the model contract')
    result = neutral(n)
    known = np.zeros((128, 16), bool)
    if obj is not None:
        values, known = compile_plan(obj, frames=n)
        result = np.where(known, values, result)
    return (result.astype(np.float32), known, dict(predicted_frames=predicted, canonicalized=canonicalized, failed=error is not None, error=error))
