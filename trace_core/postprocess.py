"""离线零相位平滑：保持首尾姿态，不改变动作时长，不引入 IK。"""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation


def lowpass(values, fps, cutoff_hz, order=4):
    """双向 Butterworth 滤波，并用线性补偿消除首尾偏差。"""
    if not 0 < cutoff_hz < fps / 2 or not 1 <= order <= 8:
        raise ValueError('截止频率必须位于 (0, FPS/2)，滤波阶数必须位于 [1,8]')
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 4:
        return values.copy()
    sos = butter(order, cutoff_hz, fs=fps, btype='lowpass', output='sos')
    try:
        result = sosfiltfilt(sos, values, axis=0)
    except ValueError:
        result = sosfiltfilt(sos, values, axis=0, padlen=min(len(values)-1, max(3, order*2)))
    alpha = np.linspace(0, 1, len(values)).reshape((-1,) + (1,) * (values.ndim-1))
    result += (1-alpha)*(values[0]-result[0]) + alpha*(values[-1]-result[-1])
    return result


def acceleration_energy(values):
    """用于检查平滑前后高频变化；不是物理可执行性评分。"""
    return float(np.mean(np.diff(values, n=2, axis=0)**2)) if len(values) > 2 else 0.0


def smooth_pose(pose, root_hz=3.0, joint_hz=4.0, order=4, fps=20):
    """对解码后的轨迹滤波，避免对位移增量滤波后重新积分引起漂移。"""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 35 or not len(pose) or not np.isfinite(pose).all():
        raise ValueError('需要有限的 [T,35] SPS 姿态数组')
    result = pose.copy()
    result[:, :3] = lowpass(pose[:, :3], fps, root_hz, order)
    # 先转换到标准欧拉支路并展开，避免转身经过 ±180 度时出现人为跳变。
    angles = Rotation.from_euler('xyz', pose[:, 3:6], degrees=True).as_euler('xyz')
    angles = np.unwrap(angles, axis=0)
    filtered = lowpass(angles, fps, root_hz, order)
    result[:, 3:6] = Rotation.from_euler('xyz', filtered).as_euler('xyz', degrees=True)
    joints = np.unwrap(np.deg2rad(pose[:, 6:]), axis=0)
    result[:, 6:] = np.rad2deg(lowpass(joints, fps, joint_hz, order))
    # G1 关节使用连续的有限角度，不对跨越整圈的预测静默修正。
    if np.max(np.abs(np.rad2deg(joints)-pose[:, 6:])) > 180:
        raise ValueError('关节存在跨整圈跳变，请检查原始输出；拒绝生成看似正常的平滑结果')
    result[[0, -1]] = pose[[0, -1]]
    report = dict(method='zero_phase_butterworth_endpoint_preserving', fps=fps,
        root_cutoff_hz=root_hz, joint_cutoff_hz=joint_hz, order=order,
        joint_acceleration_before=acceleration_energy(joints),
        joint_acceleration_after=acceleration_energy(np.deg2rad(result[:, 6:])),
        root_acceleration_before=acceleration_energy(pose[:, :3]*.01),
        root_acceleration_after=acceleration_energy(result[:, :3]*.01),
        endpoint_max_error=float(np.max(np.abs(result[[0, -1]]-pose[[0, -1]]))),
        near_euler_singularity=bool((np.abs(np.cos(angles[:, 1])) < .05).any()),
        ik=False)
    if not np.isfinite(result).all():
        raise ValueError('平滑产生非有限值')
    return result.astype(np.float32), report
