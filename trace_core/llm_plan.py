# 本模块由当前推理实现抽取；仅整理依赖和中文注释，计算逻辑保持一致。
import json
import math
import numpy as np
GROUPS = [(2, 5), (5, 8), (8, 12), (12, 16)]
SYSTEM = 'Convert the user motion description into a sparse G1 motion event plan. Return only JSON, no explanation. Schema: {"frames":N,"events":[[end,left_support,right_support,root_vertical,root_height,left_hand,right_hand],...]}. N is an integer from 1 to 512 at 20 FPS. Each event run ends at exclusive patch index end, with 4 frames per patch. Runs start at zero, have strictly increasing end indices, and cover ceil(N/4) patches without gaps. Merge identical consecutive states. Support: 0=not stable support, 1=stable support. Root vertical: 0=descending, 1=level, 2=ascending. Root height above ground: 0=below0.60m, 1=0.60to0.85m, 2=above0.85m. Hand height relative to pelvis: 0=below0m, 1=0to0.25m, 2=0.25to0.45m, 3=above0.45m. Use -1 for an uncertain event state. Lack of stable support does not necessarily mean airborne. Predict a coherent realization of the description, not a list of alternative motions.'

def encode_target(n, contact, cw, extra, ew):
    patches = (int(n) + 3) // 4
    states = np.full((patches, 6), -1, np.int64)
    for j in range(2):
        p = contact[:patches, j]
        good = (cw[:patches, j] >= 0.5) & ((p <= 0.25) | (p >= 0.75))
        states[good, j] = (p[good] >= 0.5).astype(np.int64)
    for j, (a, b) in enumerate(GROUPS):
        a -= 2
        b -= 2
        y = extra[:patches, a:b]
        good = (ew[:patches, a] >= 0.5) & (y.max(-1) >= 0.6)
        states[good, j + 2] = y[good].argmax(-1)
    ends = np.r_[np.flatnonzero(np.any(states[1:] != states[:-1], axis=1)) + 1, patches]
    rows = [[int(end), *states[end - 1].tolist()] for end in ends]
    return dict(frames=int(n), events=rows)

def parse(text):
    """严格检查时长、事件类别与连续覆盖；非法 Plan 必须显式报错。"""
    obj = json.loads(text)
    if not isinstance(obj, dict) or set(obj) != set(['frames', 'events']):
        raise ValueError('Expected frames/events object')
    n = obj['frames']
    events = obj['events']
    if type(n) != int or not 1 <= n <= 512:
        raise ValueError('Invalid frame count')
    if not isinstance(events, list) or not 1 <= len(events) <= (n + 3) // 4:
        raise ValueError('Invalid event count')
    last = 0
    previous = None
    for row in events:
        if not isinstance(row, list) or len(row) != 7 or any((type(v) != int for v in row)):
            raise ValueError('Expected seven integer event entries')
        end, *state = row
        if not last < end <= (n + 3) // 4:
            raise ValueError('Invalid run endpoint')
        if any((not -1 <= s < k for s, k in zip(state, [2, 2, 3, 3, 4, 4]))):
            raise ValueError('Invalid category')
        if state == previous:
            raise ValueError('Adjacent identical runs are not canonical')
        last = end
        previous = state
    if last != (n + 3) // 4:
        raise ValueError('Incomplete timeline')
    return obj

def compile_plan(obj, frames=None):
    """将游程事件转为逐 patch 概率和已知掩码；未知不等于负类别。"""
    obj = parse(json.dumps(obj))
    old = (obj['frames'] + 3) // 4
    n = obj['frames'] if frames is None else int(frames)
    if not 1 <= n <= 512:
        raise ValueError('Invalid requested length')
    dense = np.zeros((old, 16), np.float32)
    known = np.zeros_like(dense, dtype=bool)
    start = 0
    for end, lc, rc, vertical, height, lh, rh in obj['events']:
        for j, s in enumerate([lc, rc]):
            if s >= 0:
                dense[start:end, j] = s
                known[start:end, j] = True
        for (a, b), s in zip(GROUPS, [vertical, height, lh, rh]):
            if s >= 0:
                dense[start:end, a + s] = 1
                known[start:end, a:b] = True
        start = end
    patches = (n + 3) // 4
    indices = np.minimum(((np.arange(patches) + 0.5) * old / patches).astype(int), old - 1)
    p = np.zeros((128, 16), np.float32)
    mask = np.zeros_like(p, dtype=bool)
    p[:patches] = dense[indices]
    mask[:patches] = known[indices]
    return (p, mask)
