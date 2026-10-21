"""Experimental numerical jump targets applied inside reference resampling.

This is inference-time differentiable FK guidance, not a newly trained editor
or a dynamics solver. Height is measured against the fixed z=0 ground plane.
"""
import math
import torch
from .editor_geometry import G1SoleForwardKinematics


class JumpHeightGuidance:
    def __init__(self, source, mean, std, editable, active_span, gain_cm, fps=20):
        if not 0 < gain_cm <= 25:
            raise ValueError("Experimental jump gains must be in (0, 25] cm")
        self.source = source.detach().clone()
        self.mean = torch.as_tensor(mean, device=source.device).float()[None, None]
        self.std = torch.as_tensor(std, device=source.device).float()[None, None]
        self.editable = editable.detach().clone()
        self.start, self.end = active_span
        self.fps = fps
        self.fk = G1SoleForwardKinematics().to(source.device)
        self.valid = torch.ones(source.shape[:2], dtype=torch.bool, device=source.device)
        self.raw_source = self.source*self.std+self.mean
        with torch.no_grad():
            self.source_points = self.fk(self.raw_source, self.valid)
            clearance = self.source_points[0, ..., 2].amin(dim=(1, 2))
            self.peak = int(clearance[self.start:self.end].argmax())+self.start
            if float(clearance[self.peak]) < .02:
                raise ValueError("A detectable source jump is required; no jump was found")
            self.target = float(clearance[self.peak])+gain_cm/100
            self.root_target = float(self.raw_source[0, self.start:self.end, 2].max()/100)+gain_cm/100
            times = (torch.arange(source.shape[1], device=source.device)-self.peak)/fps
            # A short parabolic clearance envelope, not a claim of COM dynamics.
            self.envelope = (self.target-.5*9.81*times.square()).clamp_min(0)
            self.root_envelope = self.root_target-.5*9.81*times.square()
            self.stance = self.envelope[self.start:self.end] == 0
            self.flight = ~self.stance
            if self.peak-math.sqrt(2*self.target/9.81)*fps <= self.start+2:
                raise ValueError("The edit interval is too short for the requested jump")
        self.report = dict(
            type="in_denoising_differentiable_FK_jump_target",
            gain_cm=gain_cm, source_apex_frame=self.peak,
            target_both_feet_clearance_cm=self.target*100,
            target_root_peak_cm=self.root_target*100,
            fixed_ground_m=0., active_span=list(active_span), fps=fps,
            constraints=["sole clearance", "root apex", "stance contact",
                         "penetration", "temporal continuity", "reference preservation"],
            physical_rollout_verified=False,
        )

    def __call__(self, clean, progress, final=False):
        if progress < .65:
            return clean
        # The surrounding sampler uses inference mode. Clone into normal tensors
        # explicitly so autograd is limited to this small trajectory optimization.
        with torch.inference_mode(False), torch.enable_grad():
            initial = clean.detach().clone()
            value = initial.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([value], lr=.025 if final else .018)
            a, b = self.start, self.end
            for _ in range(240 if final else 6):
                optimizer.zero_grad(set_to_none=True)
                current = torch.where(self.editable, value, self.source)
                raw = current*self.std+self.mean
                feet = self.fk(raw, self.valid)
                z = feet[0, a:b, ..., 2]
                minimum = z.amin(dim=-1)
                target = self.envelope[a:b, None]
                height_loss = ((minimum-target)/.02).square().mean()
                root_loss = ((raw[0, a:b, 2][self.flight]/100
                              -self.root_envelope[a:b][self.flight])/.03).square().mean()
                penetration = (torch.relu(-z)/.005).square().mean()
                stance_xy = (feet[0, a:b, ..., :2]-self.source_points[0, a:b, ..., :2])
                contact_loss = (stance_xy[self.stance]/.03).square().mean()
                proximity = (current-initial)[self.editable].square().mean()
                reference = (current-self.source)[self.editable].square().mean()
                # Penalize acceleration of the change, including both boundaries.
                correction = current-self.source
                acceleration = torch.diff(correction, n=2, dim=1)
                smoothness = acceleration[:, a-2:b].square().mean()
                root_acceleration = torch.diff(raw[..., 2]/100, n=2, dim=1)*self.fps**2
                acceleration_guard = (torch.relu(root_acceleration[:, a-2:b].abs()-20)/10).square().mean()
                joint_acceleration = torch.diff(raw[..., 11:23], n=2, dim=1)[:, a-2:b]
                leg_smoothness = (joint_acceleration/5).square().mean()
                loss = (3*height_loss+3*root_loss+4*penetration+.3*contact_loss
                        +.15*proximity+.03*reference+15*smoothness
                        +2*acceleration_guard+.4*leg_smoothness)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([value], 10.)
                optimizer.step()
                with torch.no_grad():
                    value.copy_(torch.where(self.editable, value, self.source))
            output = torch.where(self.editable, value, self.source).detach()
            if final:
                self.report["final_objective"] = float(loss.detach())
            return output
