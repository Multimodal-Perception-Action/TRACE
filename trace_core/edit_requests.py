"""Compile explicit relative hand-height requests without accessing target motion.

This deliberately supports a narrow grammar, not unrestricted language or motion
event grounding. Ambiguous, negated, or unsupported temporal requests are rejected.
"""
import math
import re
import numpy as np
from .edit_scope import compile_scope


def temporal_span(instruction, frames):
    """Resolve explicit fractional intervals and zero-based half-open frame spans."""
    if type(frames) is not int or not 1 <= frames <= 104:
        raise ValueError('Expected a valid source frame count, at most 104')
    text = instruction.lower()
    matches = []
    fractions = {'half': (1, 2), 'third': (1, 3), 'quarter': (1, 4)}
    for match in re.finditer(r'\b(first|second|middle|last|final)\s+(half|third|quarter)\b', text):
        position, size = match.groups()
        unit = fractions[size][0] / fractions[size][1]
        lo, hi = {'first': (0., unit), 'second': (unit, 2*unit),
                  'middle': ((1-unit)/2, (1+unit)/2), 'last': (1-unit, 1.),
                  'final': (1-unit, 1.)}[position]
        matches.append((match.start(), match.end(), math.floor(lo*frames+1e-6), math.floor(hi*frames+1e-6)))
    for match in re.finditer(r'\bzero-based\s+frames\s*\[\s*(\d+)\s*,\s*(\d+)\s*\)', text):
        matches.append((match.start(), match.end(), int(match[1]), int(match[2])))
    if len(matches) != 1:
        raise ValueError('Exactly one supported explicit temporal interval is required')
    offset, _, start, end = matches[0]
    prefix = text[max(0, offset-70):offset]
    if re.search(r'\b(except|excluding|outside|not|before|after|until)(?:\s+(?:the|only|during|within|in))*\s*$', prefix):
        raise ValueError('Negated or relative-to-interval syntax is not supported')
    if not 0 <= start < end <= frames or end-start < 2:
        raise ValueError('Temporal interval is empty or out of bounds')
    return [start, end]


def compile_request(parsed, frames):
    """Return a body-time scope and signed per-frame wrist-height request in meters."""
    scope = compile_scope(parsed, 'typed_change', frames)
    amount = float(parsed['delta_cm'])
    if not 0 < abs(amount) <= 10:
        raise ValueError('Supported relative offsets are nonzero and at most 10 cm')
    start, end = parsed['time_span']
    if end-start < 2:
        raise ValueError('At least two frames are needed for the transition profile')
    profile = np.sin(np.linspace(0, np.pi, end-start))**2
    profile[np.abs(profile)<1e-8] = 0
    if not np.any(profile):
        raise ValueError('The requested interval must include a nonzero transition interior')
    request = np.zeros((104, 2), np.float32)
    side = 0 if parsed['effector'] == 'left_hand' else 1
    request[start:end, side] = profile * (amount*.01)
    return scope, request


def resolve_typed_time(parsed, instruction, frames):
    """Retain LLM semantic fields, compile explicit time algebra from user text."""
    if not isinstance(parsed, dict) or set(parsed) != {'attribute', 'effector', 'delta_cm', 'time_span'}:
        raise ValueError('Invalid change-only semantic schema')
    updated = dict(parsed, time_span=temporal_span(instruction, frames))
    scope, request = compile_request(updated, frames)
    return updated, scope, request


def explicit_training_request(instruction, frames):
    """Parse the restricted train/validation template language, never target fields.

    Deployment uses the LLM semantic delta plus resolve_typed_time instead. This
    helper intentionally rejects multiple sides, directions or magnitudes.
    """
    text = instruction.lower()
    if re.search(r"\b(?:not|never|except|excluding|instead|don't|cannot|can't)\b", text):
        raise ValueError('Negation and exceptions require a different request')
    if re.search(r"\b(?:then|both|jump|walk|kick|foot|feet|leg|legs|torso|head)\b", text):
        raise ValueError('The template parser accepts one hand-height edit only')
    sides = set(re.findall(r'\b(left|right)\s+(?:hand|arm)\b', text))
    directions = set(re.findall(r'\b(higher|lower|raise)\b', text))
    signs = {1 if value in ('higher', 'raise') else -1 for value in directions}
    numbers = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
               'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
    amounts = []
    for value in re.findall(r'\b(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:cm|centimeters?)\b', text):
        amounts.append(float(numbers[value]) if value in numbers else float(value))
    if len(sides) != 1 or len(signs) != 1 or len(amounts) != 1:
        raise ValueError('Unsupported or ambiguous explicit hand-height instruction')
    parsed = dict(attribute='hand_height', effector=next(iter(sides))+'_hand',
                  delta_cm=next(iter(signs))*amounts[0], time_span=temporal_span(text, frames))
    scope, request = compile_request(parsed, frames)
    return parsed, scope, request
