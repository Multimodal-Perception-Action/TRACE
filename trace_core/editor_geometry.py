"""Inference-only gravity, sole FK and source-contact operations."""
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
FEATURE_DIM = 40

def rot6d_to_matrix_torch(value: Tensor) -> Tensor:
    a1, a2 = value[..., :3], value[..., 3:6]
    b1 = F.normalize(a1.float(), dim=-1, eps=1e-6)
    b2 = a2.float() - (b1 * a2.float()).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-6)
    return torch.stack((b1, b2, torch.cross(b1, b2, dim=-1)), dim=-1)


def integrate_gravity_gauge_root(motion40: Tensor, frame_mask: Tensor) -> dict[str, Tensor]:
    """Differentiably reconstruct canonical root trajectory and orientation."""
    if motion40.ndim != 3 or motion40.shape[-1] != FEATURE_DIM:
        raise ValueError(f"expected [B,T,{FEATURE_DIM}] motion")
    if frame_mask.shape != motion40.shape[:2]:
        raise ValueError("frame mask shape differs from motion")
    valid = frame_mask.to(motion40.dtype)
    yaw_vector = motion40[..., 3:5]
    yaw_valid = torch.linalg.vector_norm(yaw_vector, dim=-1) > 1e-6
    yaw_sine = torch.where(yaw_valid, yaw_vector[..., 0], torch.zeros_like(yaw_vector[..., 0]))
    yaw_cosine = torch.where(yaw_valid, yaw_vector[..., 1], torch.ones_like(yaw_vector[..., 1]))
    delta_yaw = torch.atan2(yaw_sine, yaw_cosine) * valid
    yaw = torch.cat(
        (torch.zeros_like(delta_yaw[:, :1]), torch.cumsum(delta_yaw[:, :-1], dim=1)),
        dim=1,
    )
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    local_x, local_y = motion40[..., 0], motion40[..., 1]
    world_step = torch.stack(
        (cosine * local_x - sine * local_y, sine * local_x + cosine * local_y),
        dim=-1,
    ) * valid[..., None]
    xy = torch.cat(
        (torch.zeros_like(world_step[:, :1]), torch.cumsum(world_step[:, :-1], dim=1)),
        dim=1,
    )
    position = torch.cat((xy, motion40[..., 2:3]), dim=-1)
    heading = torch.zeros(
        motion40.shape[:2] + (3, 3), device=motion40.device, dtype=torch.float32
    )
    heading[..., 0, 0] = cosine.float()
    heading[..., 1, 0] = sine.float()
    heading[..., 0, 1] = -sine.float()
    heading[..., 1, 1] = cosine.float()
    heading[..., 2, 2] = 1.0
    tilt = rot6d_to_matrix_torch(motion40[..., 5:11])
    rotation = heading @ tilt
    return {
        "position": position.float(),
        "yaw": yaw.float(),
        "tilt": tilt,
        "rotation": rotation,
        "body_up": rotation[..., :, 2],
    }


def _quaternion_wxyz_to_matrix(value: Tensor) -> Tensor:
    value = F.normalize(value.float(), dim=-1)
    w, x, y, z = value.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(value.shape[:-1] + (3, 3))


def _principal_axis_rotation(angle: Tensor, axis: int) -> Tensor:
    """Batched right-handed rotation around x/y/z (axis 0/1/2)."""
    cosine, sine = torch.cos(angle.float()), torch.sin(angle.float())
    one, zero = torch.ones_like(cosine), torch.zeros_like(cosine)
    if axis == 0:
        rows = (one, zero, zero, zero, cosine, -sine, zero, sine, cosine)
    elif axis == 1:
        rows = (cosine, zero, sine, zero, one, zero, -sine, zero, cosine)
    elif axis == 2:
        rows = (cosine, -sine, zero, sine, cosine, zero, zero, zero, one)
    else:
        raise ValueError(f"invalid principal axis: {axis}")
    return torch.stack(rows, dim=-1).reshape(angle.shape + (3, 3))


class G1SoleForwardKinematics(nn.Module):
    """Differentiable FK for the two six-DoF G1 legs.

    Constants mirror ``g1_mocap_29dof.xml``.  The four points per foot are the
    bottoms of the four small sole collision spheres on ankle-roll links.
    """

    def __init__(self, root_position_scale: float = 100.0) -> None:
        super().__init__()
        if root_position_scale <= 0:
            raise ValueError("root position scale must be positive")
        self.root_position_scale = float(root_position_scale)
        offsets = torch.tensor(
            [
                [
                    [0.0, 0.064452, -0.1027],
                    [0.0, 0.052, -0.030465],
                    [0.025001, 0.0, -0.12412],
                    [-0.078273, 0.0021489, -0.17734],
                    [0.0, -0.000094445, -0.30001],
                    [0.0, 0.0, -0.017558],
                ],
                [
                    [0.0, -0.064452, -0.1027],
                    [0.0, -0.052, -0.030465],
                    [0.025001, 0.0, -0.12412],
                    [-0.078273, -0.0021489, -0.17734],
                    [0.0, 0.000094445, -0.30001],
                    [0.0, 0.0, -0.017558],
                ],
            ],
            dtype=torch.float32,
        )
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
        hip_static = torch.tensor([0.996179, 0.0, -0.0873386, 0.0])
        knee_static = torch.tensor([0.996179, 0.0, 0.0873386, 0.0])
        static_quaternions = torch.stack(
            (identity, hip_static, identity, knee_static, identity, identity)
        )
        static_rotations = _quaternion_wxyz_to_matrix(static_quaternions)
        static_rotations = static_rotations.unsqueeze(0).repeat(2, 1, 1, 1)
        sole_points = torch.tensor(
            [
                [-0.05, 0.025, -0.035],
                [-0.05, -0.025, -0.035],
                [0.12, 0.03, -0.035],
                [0.12, -0.03, -0.035],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("offsets", offsets, persistent=False)
        self.register_buffer("static_rotations", static_rotations, persistent=False)
        self.register_buffer("sole_points", sole_points, persistent=False)
        self.axes = (1, 0, 2, 1, 1, 0)

    def forward(self, motion40: Tensor, frame_mask: Tensor) -> Tensor:
        """Return sole points with shape ``[B,T,2 feet,4 points,3 xyz]``."""
        return self.landmarks(motion40, frame_mask)["feet"]

    def landmarks(self, motion40: Tensor, frame_mask: Tensor) -> dict[str, Tensor]:
        """Return sole corners, knee centers and heading-relative torso pitch."""
        if motion40.ndim != 3 or motion40.shape[-1] != FEATURE_DIM:
            raise ValueError(f"expected [B,T,{FEATURE_DIM}] gravity-gauge motion")
        if frame_mask.shape != motion40.shape[:2]:
            raise ValueError("frame mask shape differs from motion")
        root = integrate_gravity_gauge_root(motion40.float(), frame_mask)
        # SPS stores root translations in centimeters while the G1 MJCF link
        # offsets below are meters. V26 already uses the same factor in its
        # integrated-position loss (default 100).
        root_position = root["position"] / self.root_position_scale
        root_rotation = root["rotation"]
        leg_angles = torch.stack(
            (motion40[..., 11:17], motion40[..., 17:23]), dim=2
        ).float()
        # SPS joint rotations are degrees; MuJoCo hinge coordinates and the
        # trigonometric FK below are radians.
        leg_angles = torch.deg2rad(leg_angles)
        feet: list[Tensor] = []
        knees: list[Tensor] = []
        for side in range(2):
            position = root_position
            rotation = root_rotation
            for joint in range(6):
                offset = self.offsets[side, joint]
                position = position + torch.matmul(
                    rotation, offset.reshape(1, 1, 3, 1)
                ).squeeze(-1)
                if joint == 3:
                    knees.append(position)
                rotation = torch.matmul(
                    rotation, self.static_rotations[side, joint]
                )
                rotation = torch.matmul(
                    rotation,
                    _principal_axis_rotation(
                        leg_angles[..., side, joint], self.axes[joint]
                    ),
                )
            points = position[..., None, :] + torch.einsum(
                "btij,pj->btpi", rotation, self.sole_points
            )
            feet.append(points)
        torso = root_rotation
        for index, axis in ((23, 2), (24, 0), (25, 1)):
            torso = torso @ _principal_axis_rotation(torch.deg2rad(motion40[..., index]), axis)
        up = torso[..., :, 2]
        forward_up = up[..., 0]*torch.cos(root["yaw"])+up[..., 1]*torch.sin(root["yaw"])
        pitch = torch.rad2deg(torch.atan2(forward_up, up[..., 2]))
        return dict(feet=torch.stack(feet, dim=2), knees=torch.stack(knees, dim=2), torso_pitch_deg=pitch)

def _masked_feature_mean(value: Tensor, mask: Tensor) -> Tensor:
    weight = mask.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    expanded = weight.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1)


def _sequence_ground_height(
    sole_height: Tensor, frame_mask: Tensor, quantile: float,
) -> Tensor:
    """Robust per-sequence ground estimate from clean sole heights."""
    batch = sole_height.shape[0]
    valid = frame_mask[..., None].expand_as(sole_height)
    flattened = sole_height.masked_fill(~valid, torch.inf).reshape(batch, -1)
    sorted_height = flattened.sort(dim=1).values
    counts = valid.reshape(batch, -1).sum(dim=1).clamp_min(1)
    indices = torch.floor((counts - 1).float() * quantile).long()
    return sorted_height.gather(1, indices[:, None]).squeeze(1)


def infer_clean_foot_contacts(
    sole_points: Tensor,
    frame_mask: Tensor,
    *,
    fps: float,
    height_threshold_m: float,
    speed_threshold_mps: float,
    ground_quantile: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Infer conservative clean contacts and return contact/ground/speed."""
    if sole_points.ndim != 5 or sole_points.shape[2:4] != (2, 4):
        raise ValueError("sole points must have shape [B,T,2,4,3]")
    center = sole_points.mean(dim=3)
    sole_height = sole_points[..., 2].amin(dim=3)
    ground = _sequence_ground_height(sole_height, frame_mask, ground_quantile)
    speed = torch.zeros_like(sole_height)
    if sole_points.shape[1] > 1:
        pair_speed = torch.linalg.vector_norm(
            center[:, 1:, :, :2] - center[:, :-1, :, :2], dim=-1
        ) * fps
        speed[:, 1:] = pair_speed
        speed[:, 0] = pair_speed[:, 0]
    contact = (
        (sole_height - ground[:, None, None] <= height_threshold_m)
        & (speed <= speed_threshold_mps)
        & frame_mask[..., None]
    )
    # Reject single-frame contacts; all loss terms then operate on stable
    # stance evidence rather than ambiguous landing/take-off boundaries.
    previous = F.pad(contact[:, :-1], (0, 0, 1, 0), value=False)
    following = F.pad(contact[:, 1:], (0, 0, 0, 1), value=False)
    contact = contact & (previous | following)
    return contact, ground, speed
