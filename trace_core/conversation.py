"""Persistent edit context, grounded source summaries, and validated edit intents."""
import json
import math
from pathlib import Path
import re


SYSTEM = """You interpret follow-up requests for a G1 motion-editing session.
The user message contains structured context, measured current-motion geometry,
previous requests, and the new instruction. Use the latest actual motion as the
reference. The latest caption describes the last intended action; measurements
describe the actual result. Never treat a previous target as a measured result.
Return only one JSON object with exactly these keys:
{"action":"jump_height|crouch_depth|lunge_depth|kick_height|knee_height|bow_angle|body_style|unsupported", "target_caption":"English full action description",
 "body":"lower_body|left_leg|right_leg|torso|whole_body",
 "direction":"higher|lower|deeper|shallower|none", "magnitude":"slight|moderate|large",
 "amount_cm":null, "amount_deg":null, "time_region":"full|first_half|middle_half|last_half", "reason":"short explanation"}.
For a relative airborne jump-height change use jump_height, lower_body, and
higher or lower. 'A little lower' after a jump-height edit means lower the jump,
not crouch lower. 'Again' refers to the current motion, not the initial motion.
amount_cm must be null unless the NEW instruction explicitly specifies a distance;
if specified, use its positive magnitude in cm. Do not invent numerical targets.
Use slight for a little/slightly, large for much/a lot, moderate otherwise.
Use crouch_depth for making an existing squat/crouch deeper or shallower:
body=lower_body and direction=deeper or shallower. The measured source pelvis
trajectory anchors the relative change. A deeper crouch lowers the pelvis;
a shallower crouch raises it. For a specified distance use amount_cm; otherwise
use null. 'Shallower' alone has moderate magnitude, not slight; only explicit
words such as 'a little' or 'slightly' select slight. Do not use this action
for jumping, kicking, bowing or changing to a different action.
Use lunge_depth for deeper/shallower existing lunges, body=lower_body.
Use kick_height for higher/lower kicks and knee_height for higher/lower knee lifts;
body must be left_leg or right_leg according to the source and request.
'Lift less' means lower, 'lift more' means higher. Use bow_angle for deeper/shallower
bows, body=torso. amount_deg is null unless the new request explicitly specifies
an angle in degrees; amount_cm must be null for bow_angle. All other actions use
amount_deg=null. The compiler handles necessary pelvis/hip dependencies for bows.
Use body_style only for other qualitative requests, direction=none, both amounts=null.
A jump-height edit keeps arms and torso fixed.
Preserve the action category, side, arm motion, travel and other unspecified
attributes in target_caption. Write a self-contained target caption based on
the current context and new request. Remove obsolete modifiers when reversing
a previous change. Never interpret preserving arms as a request to edit arms.
For body_style the target_caption is the actual conditioning input. It must
explicitly describe the requested change in height, depth, angle or amplitude;
repeating the original caption omits the edit. Preservation clauses must not
displace the requested change. Do not put the change only in the reason field.
Only unchanged duration, global travel, heading and endpoint poses are supported.
Requests to change timing, speed, duration, action order, robot, travel, heading,
an arm/hand pose, or to perform a different action are unsupported in this interface.
Unsupported or ambiguous requests must use action=unsupported and explain what
clarification or unsupported capability is needed. Do not silently drop part of
a compound request. Do not invent a substitute supported task.
Example: a jumping-jack source followed by 'Can you jump higher?' maps to
action=jump_height, body=lower_body, direction=higher, magnitude=moderate,
amount_cm=null. Jumping is a whole-body activity, but only root height and legs
are released by this height edit; therefore body=whole_body is incorrect here.
"""

BODIES = {"lower_body", "left_leg", "right_leg", "torso", "whole_body"}
REGIONS = {"full", "first_half", "middle_half", "last_half"}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_intent(response):
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate intent key: " + key)
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=unique)
    # Angle requests may omit the inapplicable length field, not the angle field.
    if isinstance(value, dict) and value.get('action') == 'bow_angle' and 'amount_deg' in value:
        value.setdefault('amount_cm', None)
    keys = {"action", "target_caption", "body", "direction", "magnitude", "amount_cm", "time_region", "reason"}
    if not isinstance(value, dict) or set(value) not in (keys, keys | {"amount_deg"}):
        raise ValueError("The language response does not match the edit schema")
    if value["action"] not in {"jump_height", "crouch_depth", "lunge_depth", "kick_height", "knee_height", "bow_angle", "body_style", "unsupported"}:
        raise ValueError("Unknown editing action")
    if value["body"] not in BODIES or value["time_region"] not in REGIONS:
        raise ValueError("Unknown body/time region")
    if value["direction"] not in {"higher", "lower", "deeper", "shallower", "none"} or value["magnitude"] not in {"slight", "moderate", "large"}:
        raise ValueError("Unknown relative direction or magnitude")
    for name in ("target_caption", "reason"):
        if not isinstance(value[name], str) or not value[name].strip() or len(value[name]) > 2000:
            raise ValueError("Invalid " + name)
    amount = value["amount_cm"]
    angle = value.get("amount_deg")
    if angle is not None and (type(angle) not in (int, float) or not math.isfinite(angle) or not 0 < angle <= 15):
        raise ValueError("Explicit bow angle changes must be within (0,15] degrees")
    if value["action"] != "bow_angle" and angle is not None:
        raise ValueError("Angle units are only valid for bow_angle")
    if amount is not None and (type(amount) not in (int, float) or not math.isfinite(amount) or not 0 < amount <= 25):
        raise ValueError("Explicit height changes must be within (0, 25] cm")
    if value["action"] == "jump_height":
        if value["direction"] not in {"higher", "lower"} or value["body"] != "lower_body":
            raise ValueError("Jump edits require a signed lower-body request")
    elif value["action"] in {"crouch_depth", "lunge_depth"}:
        if value["direction"] not in {"deeper", "shallower"} or value["body"] != "lower_body":
            raise ValueError("Crouch edits require deeper/shallower and lower_body")
        if amount is not None and amount > 15:
            raise ValueError("Crouch height changes must not exceed 15 cm")
    elif value["action"] in {"kick_height", "knee_height"}:
        if value["direction"] not in {"higher", "lower"} or value["body"] not in {"left_leg", "right_leg"} or (amount is not None and amount > 15):
            raise ValueError("Leg height edits require higher/lower, one leg and at most 15 cm")
    elif value["action"] == "bow_angle":
        if value["direction"] not in {"deeper", "shallower"} or value["body"] != "torso" or amount is not None:
            raise ValueError("Bow edits require torso, deeper/shallower and angle units")
    elif value["action"] == "body_style" and (amount is not None or value["direction"] != "none"):
        raise ValueError("Qualitative body edits cannot silently consume numerical goals")
    return value


def validate_explicit_distance(intent, instruction):
    """Prevent an invented or changed numeric height from reaching the sampler."""
    if intent["action"] not in {"jump_height", "crouch_depth", "lunge_depth", "kick_height", "knee_height", "bow_angle"}:
        return
    names = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    text = instruction.lower()
    for value, name in reversed(list(enumerate(names, 1))):
        text = re.sub(r"\b" + name + r"\b", str(value), text)
    if intent["action"] == "bow_angle":
        angles = [float(x) for x in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:degrees?|deg|°)", text)]
        amount = intent.get("amount_deg")
        if (amount is not None and amount not in angles) or (len(angles) == 1 and amount != angles[0]):
            raise ValueError("amount_deg must match the explicit angle, or be null when none is given")
        if re.search(r"\d\s*(?:cm|centimeters?|meters?)\b", text):
            raise ValueError("Bow angle edits cannot silently consume a length request")
        return
    distances = []
    for value, unit in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(centimeters?|centimetres?|cm|millimeters?|millimetres?|mm|meters?|metres?|m)\b", text):
        scale = .1 if unit == "mm" or unit.startswith("milli") else 100 if unit == "m" or unit in {"meter", "meters", "metre", "metres"} else 1
        distances.append(float(value)*scale)
    if intent["amount_cm"] is not None and not any(math.isclose(intent["amount_cm"], x) for x in distances):
        raise ValueError("amount_cm must match an explicit distance in the new instruction; otherwise use null")
    if len(distances) == 1 and (intent["amount_cm"] is None or not math.isclose(intent["amount_cm"], distances[0])):
        raise ValueError("Preserve the explicitly requested distance in amount_cm")


def validate_changed_caption(intent, context):
    """Reject a style edit that forwards the unchanged source caption."""
    if intent["action"] != "body_style":
        return
    normalize = lambda value: " ".join(re.findall(r"\w+", value.lower()))
    if normalize(intent["target_caption"]) == normalize(context["latest_caption"]):
        raise ValueError("target_caption repeats the source without the requested change. "
                         "Explicitly express the requested height/depth/angle/amplitude change in target_caption, "
                         "not just in reason; preserve the other source attributes.")


def source_context(source, source_text=None):
    """Recover caption and checkpoint provenance from a generated/edit directory."""
    source = Path(source).resolve()
    if source.is_dir():
        candidates = [source / name for name in ("motion_edited.sps", "motion_final.sps")]
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise ValueError("Source directory has no completed motion")
    if not source.is_file():
        raise FileNotFoundError(source)
    metadata = source.parent / "request.json"
    request = json.loads(metadata.read_text()) if metadata.is_file() else {}
    caption = source_text or request.get("text")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("Imported motions need --source-text once; later turns reuse session context")
    if source.suffix.lower() == ".sps":
        match = re.search(r"^# input_fps:\s*(\d+(?:\.\d+)?)", source.read_text(), re.M)
        if not match or float(match[1]) != 20:
            raise ValueError("Source SPS must be tagged as 20 FPS")
    from .references import read_poses
    poses = read_poses(source)
    if not 20 <= len(poses) <= 512:
        raise ValueError("Source must contain 20..512 frames")
    return source, caption.strip(), request, poses


def summarize_motion(poses, boundary=8):
    """Measure soles using the same FK and ground plane as jump guidance."""
    import numpy as np
    import torch
    from .editor_geometry import G1SoleForwardKinematics
    from .motion_io import to_gravity_gauge
    torch.set_num_threads(4)
    raw = torch.from_numpy(to_gravity_gauge(poses))[None]
    with torch.no_grad():
        landmarks = G1SoleForwardKinematics().landmarks(raw, torch.ones(raw.shape[:2], dtype=torch.bool))
        points = landmarks['feet'][0].numpy()
    from .relative_attribute_guidance import attribute_profile
    profiles = {}
    for kind in ('kick_height', 'knee_height', 'bow_angle'):
        for side in ([0, 1] if kind != 'bow_angle' else [0]):
            key = kind if kind == 'bow_angle' else kind+('_left' if side == 0 else '_right')
            profiles[key] = attribute_profile(landmarks, kind, side)[0].numpy().astype(float).tolist()
    clearance = points[..., 2].min(axis=(1, 2)) * 100
    airborne = clearance > 2
    # Bridge a single threshold flicker, but keep distinct jumps separate.
    airborne[1:-1] |= airborne[:-2] & airborne[2:]
    boundaries = np.flatnonzero(np.diff(np.r_[False, airborne, False]))
    segments = []
    for start, end in boundaries.reshape(-1, 2):
        if end - start >= 2 and end > boundary and start < len(poses)-boundary:
            peak = int(start + clearance[start:end].argmax())
            segments.append(dict(start=int(start), end=int(end), peak=peak,
                                 clearance_cm=float(clearance[peak])))
    return dict(frames=len(poses), fps=20, boundary_frames=boundary,
                measured_both_feet_peak_cm=float(clearance.max()),
                measured_root_peak_cm=float(poses[:, 1].max()),
                measured_root_heights_cm=poses[:, 1].astype(float).tolist(),
                attribute_profiles=profiles,
                attribute_peaks={k: max(v) for k, v in profiles.items()},
                airborne_segments=segments, fixed_ground_m=0.,
                coordinate_semantics="Preserve local joint rotations; world positions may move with root")


def compile_intent(intent, summary):
    """Resolve relative intent into bounded, auditable inference parameters."""
    if intent["action"] == "unsupported":
        raise ValueError("Request needs clarification or an unsupported capability: " + intent["reason"])
    n, w = summary["frames"], summary["boundary_frames"]
    intervals = {"full": (w, n-w), "first_half": (w, n//2),
                 "middle_half": (max(w, n//4), min(n-w, 3*n//4)),
                 "last_half": (n//2, n-w)}
    lo, hi = intervals[intent["time_region"]]
    if hi-lo < 4:
        raise ValueError("The requested time window leaves too few editable frames")
    result = dict(text=intent["target_caption"], body=intent["body"], active_span=[lo, hi],
                  strength=.45, cfg=2.5, height_gain_cm=None, pelvis_delta_cm=None,
                  target_origin="language-only body style")
    if intent["action"] in {"kick_height", "knee_height", "bow_angle"}:
        from .relative_attribute_guidance import relative_peak_target
        kind = intent['action']
        side = 1 if intent['body'] == 'right_leg' else 0
        key = kind if kind == 'bow_angle' else kind+('_left' if side == 0 else '_right')
        amount = intent.get('amount_deg') if kind == 'bow_angle' else intent['amount_cm']
        if amount is None:
            amount = {'slight': 3., 'moderate': 5., 'large': 8.}[intent['magnitude']]
            origin = 'default measured relative policy (slight=3, moderate=5, large=8; cm for height, degrees for bow)'
        else:
            origin = 'explicit amount in the new instruction'
        delta = amount if intent['direction'] in {'higher', 'deeper'} else -amount
        _, _, measured = relative_peak_target(summary['attribute_profiles'][key], [lo, hi], delta, kind)
        result.update(relative_attribute=kind, attribute_delta=delta, target_origin=origin,
                      attribute_measurement=measured)
        return result
    if intent["action"] in {"crouch_depth", "lunge_depth"}:
        from .crouch_guidance import relative_crouch_target
        amount = intent["amount_cm"]
        if amount is None:
            amount = {"slight": 3., "moderate": 5., "large": 8.}[intent["magnitude"]]
            origin = "default relative pelvis strength policy (slight=3, moderate=5, large=8 cm)"
        else:
            origin = "explicit distance parsed from user instruction"
        delta = -amount if intent["direction"] == "deeper" else amount
        _, _, measured = relative_crouch_target(summary["measured_root_heights_cm"], [lo, hi], delta)
        result.update(pelvis_delta_cm=delta, target_origin=origin,
                      target_root_minimum_cm=measured["target_minimum_cm"],
                      source_root_minimum_cm=measured["source_minimum_cm"],
                      selected_source_minimum=measured["source_minimum_frame"])
        return result
    if intent["action"] != "jump_height":
        return result
    candidates = [x for x in summary["airborne_segments"] if lo < x["peak"] < hi-1]
    if len(candidates) > 1:
        ranked = sorted(candidates, key=lambda x: x["clearance_cm"], reverse=True)
        # A minor landing rebound must not compete with a clearly dominant jump.
        if ranked[0]["clearance_cm"] >= 2*ranked[1]["clearance_cm"]:
            candidates = ranked[:1]
    if len(candidates) != 1:
        raise ValueError("Specify a time region containing exactly one detectable or clearly dominant jump")
    event = candidates[0]
    amount = intent["amount_cm"]
    if amount is None:
        amount = {"slight": 5., "moderate": 10., "large": 14.}[intent["magnitude"]]
        origin = "default relative strength policy (slight=5, moderate=10, large=14 cm)"
    else:
        origin = "explicit distance parsed from user instruction"
    gain = amount if intent["direction"] == "higher" else -amount
    target = event["clearance_cm"] + gain
    if not 2 <= target <= 45:
        raise ValueError("The requested jump target is outside the supported 2..45 cm clearance range")
    peak = event["peak"]
    span = [max(lo, min(peak-10, event["start"]-4)),
            min(hi, max(peak+18, event["end"]+6))]
    half_flight = math.sqrt(2*target/100/9.81)*20
    if peak-half_flight <= span[0]+2 or peak+half_flight >= span[1]-2:
        raise ValueError("The fixed-duration source has insufficient room for this jump target")
    result.update(active_span=span, strength=.65, cfg=3., height_gain_cm=gain,
                  target_clearance_cm=target, target_origin=origin,
                  selected_source_apex=peak,
                  event_selection="single jump or dominant apex at least twice the next clearance")
    return result


def resolve_with_llm(context_path, model_path, device):
    """Use the shipped instruction-tuned Qwen base, without the event-only adapter."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from .pipeline import language_dtype
    torch.set_num_threads(4)
    context_path = Path(context_path)
    context = json.loads(context_path.read_text())
    device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        torch_dtype=language_dtype(device), attn_implementation="sdpa").to(device).eval()
    # Per-frame profiles are used by the compiler, not inflated into LLM tokens.
    language_context = dict(context, source_summary={k: v for k, v in context['source_summary'].items()
        if k not in {'attribute_profiles', 'measured_root_heights_cm'}})
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(language_context, ensure_ascii=False)}]
    for attempt in range(2):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
        tokens = tokenizer(prompt, return_tensors="pt").to(device)
        if tokens["input_ids"].shape[1] > 6000:
            raise ValueError("Conversation context exceeds the supported token budget")
        with torch.inference_mode():
            result = model.generate(**tokens, max_new_tokens=900, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(result[0, tokens["input_ids"].shape[1]:], skip_special_tokens=True)
        (context_path.parent / f"language_attempt_{attempt+1:02d}.txt").write_text(response + "\n")
        try:
            intent = parse_intent(response)
            validate_explicit_distance(intent, context["instruction"])
            validate_changed_caption(intent, context)
            break
        except (ValueError, TypeError) as error:
            if attempt == 1:
                raise
            messages += [{"role": "assistant", "content": response},
                         {"role": "user", "content": "Correct this validation error while preserving my request: " + str(error)}]
    (context_path.parent / "language_response.txt").write_text(response + "\n")
    atomic_json(context_path.parent / "intent.json", intent)
    atomic_json(context_path.parent / "compiled_edit.json", compile_intent(intent, context["source_summary"]))
