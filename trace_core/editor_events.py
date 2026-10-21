"""Source-motion palm FK and event labels shared with the trained editor."""
import numpy as np
import torch
from torch import nn
from .editor_geometry import integrate_gravity_gauge_root, _quaternion_wxyz_to_matrix
from .motion_io import SPS_JOINTS
G1_JOINT_ORDER = SPS_JOINTS[1:]
GROUPS = ((2, 5), (5, 8), (8, 12), (12, 16))
SCHEMA = dict(channels=16, patch_frames=4, fps=20, contact=['left_stable_support', 'right_stable_support'], root_vertical=['descending', 'level', 'ascending'], root_height=['low', 'middle', 'high'], left_hand=['below_pelvis', 'lower_torso', 'upper_torso', 'above_shoulder'], right_hand=['below_pelvis', 'lower_torso', 'upper_torso', 'above_shoulder'], thresholds=dict(vertical_speed_mps=[-0.08, 0.08], root_height_above_ground_m=[0.6, 0.85], hand_height_above_pelvis_m=[0.0, 0.25, 0.45], height_uncertainty_m=0.025, speed_uncertainty_mps=0.02), notes='Only support is a stable-contact label; no-support is not airborne. Five-frame moving average before event quantization.')

class PalmFK(nn.Module):
    """Batched Torch palm FK from MJCF hinge chains; no joint clipping."""

    def __init__(self, xml=None):
        super().__init__()
        import mujoco
        if xml is None:
            raise ValueError('Supply the licensed G1 MJCF with left_palm/right_palm sites')
        model = mujoco.MjModel.from_xml_path(str(xml))
        pelvis = model.body('pelvis').id
        self.chains = []
        indices = {name: i for (i, name) in enumerate(G1_JOINT_ORDER)}
        for side in ('left', 'right'):
            # The distributed G1 MJCF omits display sites. Use the same 8 cm
            # palm offset as the training FK without modifying the robot asset.
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, side + '_palm')
            if site >= 0:
                body = int(model.site_bodyid[site])
                site_position = model.site_pos[site].copy()
            else:
                body = model.body(side + '_wrist_yaw_link').id
                site_position = np.array([.08, 0., 0.])
            chain = []
            while body != pelvis:
                if body == 0:
                    raise ValueError('Palm must descend from pelvis')
                joints = []
                for j in range(int(model.body_jntadr[body]), int(model.body_jntadr[body] + model.body_jntnum[body])):
                    if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                        raise ValueError('Only hinge chains supported')
                    name = model.joint(j).name
                    joints.append((indices[name], model.jnt_axis[j].copy(), model.jnt_pos[j].copy()))
                chain.append((model.body_pos[body].copy(), model.body_quat[body].copy(), joints))
                body = int(model.body_parentid[body])
            self.chains.append((list(reversed(chain)), site_position))

    def forward(self, x, mask):
        root = integrate_gravity_gauge_root(x.float(), mask)
        result = []
        for (chain, site) in self.chains:
            pos = torch.zeros_like(x[..., :3])
            rot = root['rotation']
            for (offset, quat, joints) in chain:
                pos = pos + torch.einsum('btij,j->bti', rot, x.new_tensor(offset))
                rot = rot @ _quaternion_wxyz_to_matrix(x.new_tensor(quat))
                for (index, axis, anchor) in joints:
                    axis = x.new_tensor(axis)
                    anchor = x.new_tensor(anchor)
                    pos = pos + torch.einsum('btij,j->bti', rot, anchor)
                    theta = torch.deg2rad(x[..., 11 + index].float())
                    (ax, ay, az) = axis.unbind()
                    z = ax.new_zeros(())
                    cross = torch.stack([z, -az, ay, az, z, -ax, -ay, ax, z]).reshape(3, 3)
                    eye = torch.eye(3, device=x.device)
                    rotation = eye + theta.sin()[..., None, None] * cross + (1 - theta.cos())[..., None, None] * (cross @ cross)
                    rot = rot @ rotation
                    pos = pos - torch.einsum('btij,j->bti', rot, anchor)
            result.append(pos + torch.einsum('btij,j->bti', rot, x.new_tensor(site)))
        return torch.stack(result, dim=2)

def smooth(x):
    x = np.asarray(x, dtype=np.float32)
    return np.convolve(np.pad(x, (2, 2), mode='edge'), np.ones(5, dtype=np.float32) / 5, mode='valid')

def label_extra(root_height, hands, ground):
    """Return soft patch targets/confidence [128,14]; ambiguous frames are masked."""
    n = len(root_height)
    assert 1 <= n <= 512 and hands.shape == (n, 2)
    height = smooth(root_height - ground)
    speed = np.gradient(height) * 20 if n > 1 else np.zeros(n)
    signals = [speed, height, smooth(hands[:, 0]), smooth(hands[:, 1])]
    boundaries = [[-0.08, 0.08], [0.6, 0.85], [0.0, 0.25, 0.45], [0.0, 0.25, 0.45]]
    margins = [0.02, 0.025, 0.025, 0.025]
    target = np.zeros((128, 14), np.float32)
    weight = np.zeros_like(target)
    offset = 0
    for (values, cuts, margin) in zip(signals, boundaries, margins):
        values = np.asarray(values)
        cuts = np.asarray(cuts)
        classes = len(cuts) + 1
        labels = np.searchsorted(cuts, values, side='right')
        confidence = np.min(np.abs(values[:, None] - cuts), axis=1) >= margin
        for (patch, start) in enumerate(range(0, n, 4)):
            c = confidence[start:start + 4]
            y = labels[start:start + 4]
            if c.any():
                target[patch, offset:offset + classes] = np.bincount(y[c], minlength=classes) / c.sum()
            weight[patch, offset:offset + classes] = c.mean()
        offset += classes
    return (target, weight)
