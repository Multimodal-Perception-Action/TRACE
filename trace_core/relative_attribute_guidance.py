"""Measured source-relative leg heights and bow angles during diffusion sampling."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from .editor_geometry import G1SoleForwardKinematics, rot6d_to_matrix_torch, _principal_axis_rotation
from .motion_io import SPS_JOINTS


def attribute_profile(landmarks, kind, side=0):
    if kind == 'kick_height':
        return landmarks['feet'][..., side, :, 2].mean(-1)*100
    if kind == 'knee_height':
        return landmarks['knees'][..., side, 2]*100
    if kind == 'bow_angle':
        return landmarks['torso_pitch_deg']
    raise ValueError('Unknown relative attribute')


def relative_peak_target(profile, span, delta, kind):
    """Scale the measured event envelope, preserving timing and endpoint values."""
    values = np.asarray(profile, dtype=np.float64)
    a, b = map(int, span)
    if values.ndim != 1 or not np.isfinite(values).all() or not math.isfinite(delta) or not 0 < abs(delta) <= 15:
        raise ValueError('A finite trajectory and nonzero change of at most 15 cm/degrees are required')
    if not 2 <= a < b <= len(values)-2 or b-a < 12:
        raise ValueError('Relative editing needs protected boundaries and at least 12 active frames')
    baseline = float(np.median(np.r_[values[:a], values[b:]]))
    peak = a+int(values[a:b].argmax())
    amplitude = float(values[peak]-baseline)
    minimum = 3. if kind == 'bow_angle' else 2.
    if amplitude < minimum:
        raise ValueError(f'No measurable {kind} event: source amplitude {amplitude:.2f} is below {minimum}')
    if not a+4 <= peak < b-4:
        raise ValueError('The selected interval clips the source event peak')
    if amplitude+delta < minimum:
        raise ValueError('The requested reduction would erase or reverse the source event; use a smaller change')
    if kind == 'bow_angle' and not 0 < values[peak]+delta < 89:
        raise ValueError('The requested bow would reach or pass a horizontal torso')
    weight = np.zeros_like(values)
    weight[a:b] = np.clip((values[a:b]-baseline)/amplitude, 0, 1)
    ramp = min(6, peak-a, b-1-peak)
    fade = .5-.5*np.cos(np.linspace(0, np.pi, ramp+1))
    weight[a:a+ramp+1] *= fade
    weight[b-ramp-1:b] *= fade[::-1]
    return (values+delta*weight).astype(np.float32), weight.astype(np.float32), dict(
        attribute=kind, unit='deg' if kind == 'bow_angle' else 'cm', source_peak=float(values[peak]),
        source_peak_frame=peak, source_baseline=baseline, source_amplitude=amplitude,
        target_peak=float(values[peak]+delta), requested_change=float(delta), active_span=[a, b])


def relative_attribute_mask(n, span, kind, side):
    mask = np.zeros((n, 40), bool)
    a, b = span
    if kind == 'bow_angle':
        # Bowing beyond the existing waist flexion needs pelvis/hip cooperation.
        # Keep root translation, yaw, waist and arm rotations unchanged.
        mask[a:b, 5:23] = True
    elif kind in {'kick_height', 'knee_height'} and side in {0, 1}:
        mask[a:b, 11+6*side:17+6*side] = True
    else:
        raise ValueError('Unsupported relative-attribute scope')
    return mask


class RelativeAttributeGuidance:
    def __init__(self, source, mean, std, editable, span, kind, side, delta,
                 final_iterations=400, step_iterations=6):
        if source.ndim != 3 or source.shape[0] != 1 or source.shape[-1] != 40:
            raise ValueError('Expected one [1,T,40] motion')
        self.source = source.detach().clone()
        self.mean = torch.as_tensor(mean, device=source.device).float()[None, None]
        self.std = torch.as_tensor(std, device=source.device).float()[None, None]
        self.raw_source = self.source*self.std+self.mean
        self.mask = editable.detach().clone()
        self.kind, self.side, self.span = kind, side, span
        self.final_iterations, self.step_iterations = final_iterations, step_iterations
        expected = torch.as_tensor(relative_attribute_mask(source.shape[1], span, kind, side), device=source.device)[None]
        if not torch.equal(expected, self.mask):
            raise ValueError('Relative-attribute mask does not match the kinematic dependencies')
        self.valid = torch.ones(source.shape[:2], dtype=torch.bool, device=source.device)
        self.fk = G1SoleForwardKinematics().to(source.device)
        with torch.no_grad():
            self.source_landmarks = self.fk.landmarks(self.raw_source, self.valid)
            profile = attribute_profile(self.source_landmarks, kind, side)[0].cpu().numpy()
        target, weight, measurement = relative_peak_target(profile, span, delta, kind)
        self.target = torch.as_tensor(target, device=source.device)[None]
        self.weight = torch.as_tensor(weight, device=source.device)[None]
        self.target_feet = self.source_landmarks['feet'].clone()
        self.target_feet[..., 2].clamp_(min=0)
        xml = Path(__file__).resolve().parents[1]/'assets/unitree_g1/g1_mocap_29dof.xml'
        joints = {n.get('name'): list(map(float, n.get('range').split())) for n in ET.parse(xml).iter('joint') if n.get('range')}
        limits = torch.as_tensor(np.rad2deg([joints[n] for n in SPS_JOINTS[1:13]]), dtype=torch.float32, device=source.device)
        self.low = torch.minimum(limits[:, 0], self.raw_source[..., 11:23])
        self.high = torch.maximum(limits[:, 1], self.raw_source[..., 11:23])
        self.report = dict(type='in_denoising_source_relative_attribute', **measurement,
            side='left' if side == 0 else 'right', physical_rollout_verified=False,
            constraints=['relative measured event trajectory', 'fixed endpoint poses',
                         'protected local joint rotations', 'joint limits relative to source',
                         'temporal correction smoothness', 'sole penetration'],
            scope='pelvis tilt and both legs' if kind == 'bow_angle' else 'selected leg only')

    def project(self, value):
        current = torch.where(self.mask, value, self.source)
        raw = current*self.std+self.mean
        legs = raw[..., 11:23].clamp(self.low, self.high)
        raw = torch.cat((raw[..., :11], legs, raw[..., 23:]), -1)
        if self.kind == 'bow_angle':
            # Tilt must stay yaw-free: arbitrary 6D optimization can otherwise
            # hide heading changes in the nominally tilt-only coordinates.
            tilt = rot6d_to_matrix_torch(raw[..., 5:11])
            residual_yaw = torch.atan2(tilt[..., 1, 0], tilt[..., 0, 0])
            tilt = _principal_axis_rotation(-residual_yaw, 2) @ tilt
            six = torch.cat((tilt[..., :, 0], tilt[..., :, 1]), -1)
            raw = torch.cat((raw[..., :5], six, raw[..., 11:]), -1)
        # Do not perturb protected coordinates through normalization round trips.
        return torch.where(self.mask, (raw-self.mean)/self.std, self.source)

    def __call__(self, clean, progress, final=False):
        if progress < .55:
            return clean
        with torch.inference_mode(False), torch.enable_grad():
            initial = clean.detach().clone()
            value = initial.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([value], lr=.025 if final else .015)
            a, b = self.span
            for _ in range(self.final_iterations if final else self.step_iterations):
                optimizer.zero_grad(set_to_none=True)
                current = self.project(value)
                raw = current*self.std+self.mean
                lm = self.fk.landmarks(raw, self.valid)
                measured = attribute_profile(lm, self.kind, self.side)
                target_loss = ((measured[:, a:b]-self.target[:, a:b])/.4).square().mean()
                feet = lm['feet']
                correction = raw-self.raw_source
                temporal = (torch.diff(correction[..., 11:23], n=2, dim=1)[:, a-2:b]/3).square().mean()
                reference = (correction[..., 11:23][:, a:b]/20).square().mean()
                proximity = (current-initial)[self.mask].square().mean()
                # Preserve existing penetrations, but never encourage adding them.
                floor = torch.minimum(self.source_landmarks['feet'][..., 2], torch.zeros_like(feet[..., 2]))
                penetration = (torch.relu(floor-feet[..., 2])/.004).square().mean()
                if self.kind == 'bow_angle':
                    foot_loss = ((feet[:, a:b]-self.target_feet[:, a:b])/.008).square().mean()
                    tilt_loss = (correction[:, a:b, 5:11]/.3).square().mean()
                    tilt_smooth = (torch.diff(correction[..., 5:11], n=2, dim=1)[:, a-2:b]/.03).square().mean()
                    ancillary = 3*foot_loss+.05*tilt_loss+.3*tilt_smooth
                else:
                    foot_error = feet[:, a:b, self.side]-self.source_landmarks['feet'][:, a:b, self.side]
                    # Keep the kick's forward extent and sole orientation while changing height.
                    xy = (foot_error[..., :2]/(.06 if self.kind == 'kick_height' else .2)).square().mean()
                    shape = ((foot_error-foot_error.mean(2, keepdim=True))/.025).square().mean()
                    ancillary = .3*xy+.2*shape
                    if self.kind == 'knee_height':
                        # A higher knee request should not be realized by lowering the foot.
                        signed = math.copysign(1., self.report['requested_change'])
                        foot_delta = foot_error[..., 2].mean(-1)
                        ancillary += (torch.relu(-signed*foot_delta)/.01).square().mean()
                loss = 4*target_loss+2*temporal+.3*reference+.01*proximity+4*penetration+ancillary
                loss.backward()
                torch.nn.utils.clip_grad_norm_([value], 10.)
                optimizer.step()
                with torch.no_grad():
                    value.copy_(self.project(value))
            output = self.project(value).detach()
            if final:
                with torch.no_grad():
                    lm = self.fk.landmarks(output*self.std+self.mean, self.valid)
                    actual = attribute_profile(lm, self.kind, self.side)
                    peak = self.report['source_peak_frame']
                    self.report.update(achieved_peak=float(actual[:, a:b].max()),
                        achieved_at_source_peak=float(actual[0, peak]),
                        profile_rmse=float((actual[:, a:b]-self.target[:, a:b]).square().mean().sqrt()),
                        max_added_penetration_cm=float(torch.relu(floor-lm['feet'][..., 2]).max()*100),
                        sole_reference_rmse_cm=float((lm['feet'][:, a:b]-self.target_feet[:, a:b]).square().mean().sqrt()*100))
            return output


def validate_attribute_report(report):
    keys = ('achieved_peak', 'profile_rmse', 'max_added_penetration_cm')
    if not all(math.isfinite(report[k]) for k in keys):
        raise ValueError('Nonfinite relative-edit diagnostics')
    error = abs(report['achieved_peak']-report['target_peak'])
    change = report['achieved_peak']-report['source_peak']
    if error > .8 or report['profile_rmse'] > .8 or change*report['requested_change'] <= 0:
        raise ValueError('Relative edit did not achieve its requested direction/target; source retained')
    if report['max_added_penetration_cm'] > 1.:
        raise ValueError('Relative edit introduced excessive sole penetration; source retained')
    if report['attribute'] == 'bow_angle' and report['sole_reference_rmse_cm'] > 2.5:
        raise ValueError('Bowing could not preserve the sole reference; source retained')
