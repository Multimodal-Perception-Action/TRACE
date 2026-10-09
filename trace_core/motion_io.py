# 本模块由当前推理实现抽取；仅整理依赖和中文注释，计算逻辑保持一致。
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
SPS_JOINTS = ('root', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint')
FEATURE_DIM = 38
GRAVITY_GAUGE_FEATURE_DIM = 40
SPS_DIM = 35
_IDENTITY_6D = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
_DOF_LINE_RE = re.compile(' (tx|ty|tz|rx|ry|rz):')
_DOF_VAL_RE = re.compile('(tx|ty|tz|rx|ry|rz):([+-]?\\d+(?:\\.\\d+)?)')

def parse_sps_file(path: str | Path) -> np.ndarray:
    frames: list[dict[str, dict[str, float]]] = []
    current: dict[str, dict[str, float]] | None = None
    with Path(path).open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith('#') or line.startswith('skel'):
                continue
            if re.fullmatch('\\d+', line):
                if current is not None:
                    frames.append(current)
                current = {}
            elif current is not None and _DOF_LINE_RE.search(line):
                values = _DOF_VAL_RE.findall(line)
                if values:
                    current[line.split()[0]] = {key: float(value) for key, value in values}
    if current is not None:
        frames.append(current)
    data = np.zeros((len(frames), len(SPS_JOINTS), 6), dtype=np.float32)
    for frame_index, frame in enumerate(frames):
        for joint_index, joint in enumerate(SPS_JOINTS):
            if joint not in frame:
                continue
            value = frame[joint]
            data[frame_index, joint_index] = [value.get(k, 0.0) for k in ('tx', 'ty', 'tz', 'rx', 'ry', 'rz')]
    return data

def convert_to_35dim(data: np.ndarray) -> np.ndarray:
    if data.ndim != 3 or data.shape[1:] != (30, 6):
        raise ValueError(f'expected [T,30,6], got {data.shape}')
    output = np.zeros((len(data), SPS_DIM), dtype=np.float32)
    output[:, :6] = data[:, 0, :]
    output[:, 6:18] = data[:, 1:13, 4]
    output[:, 18:35] = data[:, 13:30, 4]
    return output

def euler_xyz_to_matrix(euler_deg: np.ndarray) -> np.ndarray:
    rpy = np.deg2rad(np.asarray(euler_deg, dtype=np.float64))
    rx, ry, rz = (rpy[..., 0], rpy[..., 1], rpy[..., 2])
    cx, sx, cy, sy, cz, sz = (np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz))
    matrix = np.empty(rpy.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = cz * cy
    matrix[..., 0, 1] = cz * sy * sx - sz * cx
    matrix[..., 0, 2] = cz * sy * cx + sz * sx
    matrix[..., 1, 0] = sz * cy
    matrix[..., 1, 1] = sz * sy * sx + cz * cx
    matrix[..., 1, 2] = sz * sy * cx - cz * sx
    matrix[..., 2, 0] = -sy
    matrix[..., 2, 1] = cy * sx
    matrix[..., 2, 2] = cy * cx
    return matrix

def matrix_to_euler_xyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    cy = np.sqrt(matrix[..., 0, 0] ** 2 + matrix[..., 1, 0] ** 2)
    ry = np.arctan2(-matrix[..., 2, 0], cy)
    rx = np.arctan2(matrix[..., 2, 1], matrix[..., 2, 2])
    rz = np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0])
    return np.rad2deg(np.stack((rx, ry, rz), axis=-1))

def rotmat_to_6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)

def rot6d_to_matrix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    a1, a2 = (value[..., :3], value[..., 3:6])
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12, None)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-12, None)
    return np.stack((b1, b2, np.cross(b1, b2)), axis=-1)

def heading_basis_from_rotation(rotation: np.ndarray) -> np.ndarray:
    forward = np.asarray(rotation, dtype=np.float64)[..., :, 0].copy()
    forward[..., 2] = 0.0
    norm = np.linalg.norm(forward, axis=-1, keepdims=True)
    fallback = np.zeros_like(forward)
    fallback[..., 0] = 1.0
    forward = np.where(norm > 1e-08, forward / np.maximum(norm, 1e-08), fallback)
    left = np.stack((-forward[..., 1], forward[..., 0], np.zeros_like(forward[..., 0])), axis=-1)
    up = np.zeros_like(forward)
    up[..., 2] = 1.0
    return np.stack((forward, left, up), axis=-1)

def heading_basis_from_yaw(yaw: np.ndarray | float) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=np.float64)
    cosine, sine = (np.cos(yaw), np.sin(yaw))
    zeros = np.zeros_like(cosine)
    ones = np.ones_like(cosine)
    return np.stack((np.stack((cosine, sine, zeros), axis=-1), np.stack((-sine, cosine, zeros), axis=-1), np.stack((zeros, zeros, ones), axis=-1)), axis=-1)

def yaw_from_rotation(rotation: np.ndarray) -> np.ndarray:
    heading = heading_basis_from_rotation(rotation)
    return np.arctan2(heading[..., 1, 0], heading[..., 0, 0])

def _sps_position_to_world(position: np.ndarray) -> np.ndarray:
    return np.asarray(position, dtype=np.float64)[..., [2, 0, 1]]

def _world_position_to_sps(position: np.ndarray) -> np.ndarray:
    return np.asarray(position, dtype=np.float64)[..., [1, 2, 0]]

def to_heading_local(data35: np.ndarray) -> np.ndarray:
    data35 = np.asarray(data35, dtype=np.float32)
    if data35.ndim != 2 or data35.shape[1] != SPS_DIM:
        raise ValueError(f'expected [T,35], got {data35.shape}')
    output = np.zeros((len(data35), FEATURE_DIM), dtype=np.float32)
    output[:, 9:] = data35[:, 6:]
    if len(data35) == 0:
        return output
    position = _sps_position_to_world(data35[:, :3])
    rotation = euler_xyz_to_matrix(data35[:, 3:6])
    output[:, 2] = position[:, 2]
    if len(data35) == 1:
        output[0, 3:9] = _IDENTITY_6D
        return output
    delta = position[1:] - position[:-1]
    heading = heading_basis_from_rotation(rotation[:-1])
    local = np.einsum('tji,tj->ti', heading, delta)
    output[:-1, :2] = local[:, :2]
    output[-1, :2] = output[-2, :2]
    relative = rotation[:-1].transpose(0, 2, 1) @ rotation[1:]
    output[:-1, 3:9] = rotmat_to_6d(relative)
    output[-1, 3:9] = output[-2, 3:9]
    return output

def from_heading_local(data38: np.ndarray) -> np.ndarray:
    data38 = np.asarray(data38, dtype=np.float32)
    if data38.ndim != 2 or data38.shape[1] != FEATURE_DIM:
        raise ValueError(f'expected [T,38], got {data38.shape}')
    output = np.zeros((len(data38), SPS_DIM), dtype=np.float32)
    output[:, 6:] = data38[:, 9:]
    if len(data38) == 0:
        return output
    position = np.zeros((len(data38), 3), dtype=np.float64)
    position[:, 2] = data38[:, 2]
    euler = np.zeros((len(data38), 3), dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    for frame in range(len(data38) - 1):
        local_step = np.asarray([data38[frame, 0], data38[frame, 1], 0.0])
        world_step = heading_basis_from_rotation(rotation) @ local_step
        position[frame + 1, :2] = position[frame, :2] + world_step[:2]
        rotation = rotation @ rot6d_to_matrix(data38[frame, 3:9])
        euler[frame + 1] = matrix_to_euler_xyz(rotation)
    output[:, :3] = _world_position_to_sps(position)
    output[:, 3:6] = euler
    return output

def to_gravity_gauge(data35: np.ndarray) -> np.ndarray:
    data35 = np.asarray(data35, dtype=np.float32)
    if data35.ndim != 2 or data35.shape[1] != SPS_DIM:
        raise ValueError(f'expected [T,35], got {data35.shape}')
    output = np.zeros((len(data35), GRAVITY_GAUGE_FEATURE_DIM), dtype=np.float32)
    output[:, 11:] = data35[:, 6:]
    if len(data35) == 0:
        return output
    position = _sps_position_to_world(data35[:, :3])
    rotation = euler_xyz_to_matrix(data35[:, 3:6])
    heading = heading_basis_from_rotation(rotation)
    yaw = yaw_from_rotation(rotation)
    tilt = heading.transpose(0, 2, 1) @ rotation
    output[:, 2] = position[:, 2]
    output[:, 5:11] = rotmat_to_6d(tilt)
    if len(data35) == 1:
        output[0, 3:5] = (0.0, 1.0)
        return output
    delta = position[1:] - position[:-1]
    local = np.einsum('tji,tj->ti', heading[:-1], delta)
    output[:-1, :2] = local[:, :2]
    output[-1, :2] = output[-2, :2]
    delta_yaw = np.arctan2(np.sin(yaw[1:] - yaw[:-1]), np.cos(yaw[1:] - yaw[:-1]))
    output[:-1, 3] = np.sin(delta_yaw)
    output[:-1, 4] = np.cos(delta_yaw)
    output[-1, 3:5] = output[-2, 3:5]
    return output

def from_gravity_gauge(data40: np.ndarray) -> np.ndarray:
    """解码 40 维状态：积分平面位移和航向，直接恢复高度、倾斜和关节角。"""
    data40 = np.asarray(data40, dtype=np.float32)
    if data40.ndim != 2 or data40.shape[1] != GRAVITY_GAUGE_FEATURE_DIM:
        raise ValueError(f'expected [T,{GRAVITY_GAUGE_FEATURE_DIM}], got {data40.shape}')
    output = np.zeros((len(data40), SPS_DIM), dtype=np.float32)
    output[:, 6:] = data40[:, 11:]
    if len(data40) == 0:
        return output
    position = np.zeros((len(data40), 3), dtype=np.float64)
    position[:, 2] = data40[:, 2]
    yaw = np.zeros(len(data40), dtype=np.float64)
    if len(data40) > 1:
        delta_yaw = np.arctan2(data40[:-1, 3], data40[:-1, 4])
        yaw[1:] = np.cumsum(delta_yaw)
        local_step = np.concatenate((data40[:-1, :2], np.zeros((len(data40) - 1, 1))), axis=-1)
        heading = heading_basis_from_yaw(yaw[:-1])
        world_step = np.einsum('tij,tj->ti', heading, local_step)
        position[1:, :2] = np.cumsum(world_step[:, :2], axis=0)
    heading = heading_basis_from_yaw(yaw)
    tilt = rot6d_to_matrix(data40[:, 5:11])
    rotation = heading @ tilt
    output[:, :3] = _world_position_to_sps(position)
    output[:, 3:6] = matrix_to_euler_xyz(rotation)
    return output

def write_sps(path: str | Path, data35: np.ndarray, fps: int | None=None) -> None:
    """输出 G1 SPS：根平移为厘米，旋转为角度，关节标量写入 ry。"""
    data35 = np.asarray(data35, dtype=np.float32)
    frames = np.zeros((len(data35), 30, 6), dtype=np.float32)
    frames[:, 0, :] = data35[:, :6]
    frames[:, 1:13, 4] = data35[:, 6:18]
    frames[:, 13:30, 4] = data35[:, 18:35]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        handle.write('skel_poseset_v01\n')
        handle.write('# format: unitree_g1_29dof_sps_v1\n')
        handle.write('# root_translation: centimeters; rotations: degrees\n')
        if fps is not None:
            handle.write(f'# input_fps: {int(fps)}\n')
        for frame_index, frame in enumerate(frames):
            handle.write(f'{frame_index}\n')
            for joint, values in zip(SPS_JOINTS, frame):
                handle.write(f'{joint} tx:{values[0]:.8f} ty:{values[1]:.8f} tz:{values[2]:.8f} rx:{values[3]:.8f} ry:{values[4]:.8f} rz:{values[5]:.8f}\n')
