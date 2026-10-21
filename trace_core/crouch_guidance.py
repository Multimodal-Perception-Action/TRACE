"""Source-relative pelvis-height targets with in-sampling leg FK optimization.

The target follows the measured source crouch envelope. This is inference-time
kinematic guidance, not a trained depth editor or a physics controller.
"""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from .editor_geometry import G1SoleForwardKinematics
from .motion_io import SPS_JOINTS


def validate_crouch_report(report):
    """Reject grossly infeasible results instead of advancing a conversation."""
    for key in ("root_profile_max_error_cm", "sole_reference_rmse_cm"):
        if not math.isfinite(report[key]):
            raise ValueError("Non-finite crouch feasibility diagnostics")
    if report["root_profile_max_error_cm"] > .1 or report["sole_reference_rmse_cm"] > 2.5:
        raise ValueError("Crouch guidance could not realize the target with acceptable sole-reference error; use a smaller change")


def relative_crouch_target(heights_cm, active_span, delta_cm):
    """Positive delta raises the pelvis; negative delta deepens the crouch."""
    height = np.asarray(heights_cm, dtype=np.float64)
    if height.ndim != 1 or not np.isfinite(height).all():
        raise ValueError("Expected a finite source root-height trajectory")
    if not math.isfinite(delta_cm) or not 0 < abs(delta_cm) <= 15:
        raise ValueError("Crouch height changes must have magnitude in (0, 15] cm")
    start, end = map(int, active_span)
    if not 2 <= start < end <= len(height)-2 or end-start < 12:
        raise ValueError("Crouch editing needs a protected boundary and at least 12 active frames")
    baseline = float(np.median(np.r_[height[:start], height[end:]]))
    peak = start + int(height[start:end].argmin())
    depth = baseline - float(height[peak])
    if depth < 5:
        raise ValueError("No source crouch of at least 5 cm was found")
    if peak < start+4 or peak >= end-4:
        raise ValueError("The selected window clips the source crouch minimum")
    if delta_cm > 0 and delta_cm > depth-3:
        raise ValueError("A shallower edit must retain at least 3 cm of crouch depth")
    target_minimum = float(height[peak]) + delta_cm
    if not 30 <= target_minimum <= 100:
        raise ValueError("The requested pelvis target is outside the 30..100 cm kinematic range")
    weight = np.zeros_like(height)
    weight[start:end] = np.clip((baseline-height[start:end])/depth, 0, 1)
    # A raised-cosine gate joins the protected endpoints with zero offset/slope.
    ramp = min(6, peak-start, end-1-peak)
    fade = .5 - .5*np.cos(np.linspace(0, np.pi, ramp+1))
    weight[start:start+ramp+1] *= fade
    weight[end-ramp-1:end] *= fade[::-1]
    target = height + delta_cm*weight
    return target.astype(np.float32), weight.astype(np.float32), dict(
        source_minimum_cm=float(height[peak]), target_minimum_cm=target_minimum,
        source_minimum_frame=peak, source_standing_height_cm=baseline,
        source_depth_cm=depth, pelvis_delta_cm=float(delta_cm), active_span=[start, end])


class CrouchDepthGuidance:
    def __init__(self, source, mean, std, editable, active_span, delta_cm, fps=20,
                 final_iterations=300, step_iterations=6):
        if source.ndim != 3 or source.shape[0] != 1 or source.shape[2] != 40:
            raise ValueError("Crouch guidance expects one [1,T,40] source")
        self.source = source.detach().clone()
        self.mean = torch.as_tensor(mean, device=source.device).float()[None, None]
        self.std = torch.as_tensor(std, device=source.device).float()[None, None]
        self.editable = editable.detach().clone()
        self.raw_source = self.source*self.std+self.mean
        self.start, self.end = active_span
        self.fps = fps
        self.final_iterations, self.step_iterations = final_iterations, step_iterations
        self.valid = torch.ones(source.shape[:2], dtype=torch.bool, device=source.device)
        self.fk = G1SoleForwardKinematics().to(source.device)
        expected = torch.zeros_like(self.editable)
        expected[:, self.start:self.end, 2] = True
        expected[:, self.start:self.end, 11:23] = True
        if not torch.equal(expected, self.editable):
            raise ValueError("Crouch guidance must edit root height and both legs only")
        target, weight, measurement = relative_crouch_target(
            self.raw_source[0, :, 2].detach().cpu().numpy(), active_span, delta_cm)
        self.target_cm = torch.as_tensor(target, device=source.device)
        self.weight = torch.as_tensor(weight, device=source.device)
        self.target_normalized = (self.target_cm-self.mean[0, 0, 2])/self.std[0, 0, 2]
        with torch.no_grad():
            self.source_feet = self.fk(self.raw_source, self.valid)
            self.target_feet = self.source_feet.clone()
            self.target_feet[..., 2].clamp_(min=0)
            self.contacts = (self.source_feet[..., 2] < .03).float()
        xml = Path(__file__).resolve().parents[1] / "assets/unitree_g1/g1_mocap_29dof.xml"
        joints = {node.get("name"): list(map(float, node.get("range").split()))
                  for node in ET.parse(xml).iter("joint") if node.get("range")}
        limits = np.rad2deg(np.array([joints[name] for name in SPS_JOINTS[1:13]]))
        limits = torch.as_tensor(limits, dtype=torch.float32, device=source.device)
        # Do not pretend to repair pre-existing violations outside the edit task.
        self.lower = torch.minimum(limits[:, 0], self.raw_source[..., 11:23])
        self.upper = torch.maximum(limits[:, 1], self.raw_source[..., 11:23])
        self.report = dict(type="in_denoising_source_relative_crouch_target", **measurement,
            fps=fps, fixed_ground_m=0., target_profile="source crouch envelope with boundary taper",
            constraints=["relative pelvis trajectory", "four-point sole reference",
                         "contact height", "penetration", "joint-range excess",
                         "temporal correction smoothness", "protected joint rotations"],
            physical_rollout_verified=False)

    def project(self, value, fraction=1.):
        current = torch.where(self.editable, value, self.source)
        root = (1-fraction)*current[..., 2]+fraction*self.target_normalized[None]
        root = torch.where(self.editable[..., 2], root, self.source[..., 2])
        return torch.cat((current[..., :2], root[..., None], current[..., 3:]), dim=-1)

    def __call__(self, clean, progress, final=False):
        if progress < .55:
            return clean
        fraction = min(1., (progress-.55)/.3)
        with torch.inference_mode(False), torch.enable_grad():
            initial = clean.detach().clone()
            value = initial.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([value], lr=.028 if final else .018)
            a, b = self.start, self.end
            for _ in range(self.final_iterations if final else self.step_iterations):
                optimizer.zero_grad(set_to_none=True)
                current = self.project(value, fraction)
                raw = current*self.std+self.mean
                feet = self.fk(raw, self.valid)
                error = feet[:, a:b]-self.target_feet[:, a:b]
                xy = (error[..., :2]/.02).square().mean()
                z = ((error[..., 2]/.008).square()*(1+3*self.contacts[:, a:b])).mean()
                penetration = (torch.relu(-feet[:, a:b, ..., 2])/.004).square().mean()
                correction = (raw-self.raw_source)[..., 11:23]
                continuity = (torch.diff(correction, n=2, dim=1)[:, a-2:b]/3).square().mean()
                velocity = (torch.diff(correction, dim=1)[:, a-1:b]/8).square().mean()
                reference = (correction[:, a:b]/25).square().mean()
                proximity = (current-initial)[self.editable].square().mean()
                legs = raw[..., 11:23]
                bounds = ((torch.relu(self.lower-legs)+torch.relu(legs-self.upper))[:, a:b]/5).square().mean()
                loss = 2*xy+3*z+4*penetration+2*continuity+.2*velocity+.06*reference+.015*proximity+.3*bounds
                loss.backward()
                torch.nn.utils.clip_grad_norm_([value], 10.)
                optimizer.step()
                with torch.no_grad():
                    value.copy_(torch.where(self.editable, value, self.source))
            output = self.project(value, fraction).detach()
            if final:
                with torch.no_grad():
                    raw = output*self.std+self.mean
                    error = self.fk(raw, self.valid)[:, a:b]-self.target_feet[:, a:b]
                    self.report.update(final_objective=float(loss.detach()),
                        achieved_minimum_cm=float(raw[0, a:b, 2].min()),
                        root_profile_max_error_cm=float((raw[0, :, 2]-self.target_cm).abs().max()),
                        sole_reference_rmse_cm=float(error.square().mean().sqrt()*100))
            return output
