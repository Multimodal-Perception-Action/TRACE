import json


SYSTEM = ('Interpret a local motion-edit request. Return only a JSON object with exactly '
    'attribute, effector, delta_cm, time_span. attribute is hand_height; effector is '
    'left_hand or right_hand; delta_cm is a nonzero signed number; time_span is two '
    'zero-based integer frame indices [start,end). Protect all other regions. '
    'Use floor for fractional bounds. Higher/lower hand references mean the initial '
    'four-frame hand height relative to the pelvis. Do not output an absolute target pose. '
    'Source event groups are left support, right support, root vertical (descending,level,ascending), '
    'root height (low,middle,high), left and right hand height '
    '(below_pelvis,lower_torso,upper_torso,above_shoulder). Unknown states are -1.')


def context(row, mode):
    result = dict(frames=row['frames'], fps=20, instruction=row['instruction'])
    if mode == 'events':
        result['source_events'] = row['source_events']
    elif mode == 'source_summary':
        result['source_hand_heights_cm'] = row['source_hand_heights_cm']
    elif mode != 'text_only':
        raise ValueError(mode)
    return result


def messages(row, mode):
    return [dict(role='system', content=SYSTEM), dict(role='user', content=json.dumps(context(row, mode), separators=(',', ':')))]
