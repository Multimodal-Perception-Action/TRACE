"""Compile a validated semantic delta into a body-time mask."""
import math
import numpy as np
GROUPS = {'root': (0, 11), 'left_leg': (11, 17), 'right_leg': (17, 23),
          'torso': (23, 26), 'left_arm': (26, 33), 'right_arm': (33, 40)}

def compile_scope(parsed, mode, n, *, expand_arm=True):
    if not isinstance(parsed, dict):
        raise ValueError('Expected a JSON object')
    if mode == 'direct':
        if set(parsed) != {'body_groups', 'frame_spans'}:
            raise ValueError('Invalid direct schema')
        groups, spans = parsed['body_groups'], parsed['frame_spans']
        if not isinstance(groups, list) or not groups or any(g not in GROUPS for g in groups):
            raise ValueError('Invalid groups')
        if len(set(groups)) != len(groups):
            raise ValueError('Duplicate groups')
        ranges = [GROUPS[g] for g in groups]
    else:
        time_key = 'time_span' if mode in ('typed_frames', 'typed_change') else 'time_fraction'
        expected = {'attribute', 'effector', 'delta_cm', time_key}
        if mode != 'typed_change':
            expected.add('preserved_groups')
        if set(parsed) != expected:
            raise ValueError('Invalid typed schema')
        if parsed['attribute'] != 'hand_height' or parsed['effector'] not in ('left_hand', 'right_hand'):
            raise ValueError('Unsupported edit')
        if isinstance(parsed['delta_cm'], bool) or not isinstance(parsed['delta_cm'], (int, float)) or not math.isfinite(parsed['delta_cm']) or parsed['delta_cm'] == 0:
            raise ValueError('Invalid delta')
        if mode in ('typed_frames', 'typed_change'):
            spans = [parsed[time_key]]
        else:
            fractions = parsed[time_key]
            if not isinstance(fractions, list) or len(fractions) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in fractions):
                raise ValueError('Invalid time fractions')
            if not 0 <= fractions[0] < fractions[1] <= 1:
                raise ValueError('Out of range fractions')
            spans = [[math.floor(v*n+1e-6) for v in fractions]]
        side = parsed['effector'].split('_')[0]
        protected = [] if mode == 'typed_change' else parsed['preserved_groups']
        if not isinstance(protected, list) or any(g not in GROUPS for g in protected) or side+'_arm' in protected:
            raise ValueError('Invalid or contradictory preservation commitments')
        ranges = [GROUPS[side+'_arm'] if expand_arm else ((30, 33) if side == 'left' else (37, 40))]
    if not isinstance(spans, list) or not spans:
        raise ValueError('Empty spans')
    mask = np.zeros((104, 40), dtype=bool)
    for span in spans:
        if not isinstance(span, list) or len(span) != 2 or any(type(v) is not int for v in span):
            raise ValueError('Integer frame spans required')
        start, end = span
        if not 0 <= start < end <= n:
            raise ValueError('Invalid frame span')
        for lo, hi in ranges:
            mask[start:end, lo:hi] = True
    if not mask.any():
        raise ValueError('Empty scope')
    return mask
