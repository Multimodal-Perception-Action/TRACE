"""多点接触实验：四角约束、前后掌标签、拉普拉斯轨迹与 IK—平滑交替。"""
from dataclasses import dataclass, asdict
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from .foot_labels import FootGeometry, runs
from .contact_window import ContactWindowProblem, WindowConfig, initial_labels


@dataclass(frozen=True)
class MultiPointConfig(WindowConfig):
    swing_weight: float = .025
    smoothing_strength: float = 2.


def clean_regions(mask):
    out = np.zeros_like(mask, dtype=bool)
    for side in range(2):
        for region in range(2):
            for a, b in runs(mask[:, side, region], 3):
                out[a:b, side, region] = True
    return out


def region_signals(points, fps):
    # SOLE 顺序：后掌两点、前掌两点。
    regions = points.reshape(len(points), 2, 2, 2, 3)
    centers = regions.mean(3)
    velocity = np.diff(centers, axis=0, prepend=centers[:1]) * fps
    if len(points) > 1:
        velocity[0] = velocity[1]
    return regions[..., 2].min(3), np.linalg.norm(velocity[..., :2], axis=-1), np.abs(velocity[..., 2])


def region_labels(points, plan, fps, eligible=None, previous=None):
    h, v, vz = region_signals(points, fps)
    if eligible is None:
        eligible = (h < .065) & (vz < .6)
    enter = (h < .025) & (v < .13) & (vz < .18)
    enter |= plan[..., None] & (h < .035) & (v < .22) & (vz < .25)
    if previous is not None:
        enter |= previous & (h < .065) & (v < .55) & (vz < .45)
    return clean_regions(enter & eligible), eligible


def target_trajectories(points, labels):
    """接触段取原动作均值；对位移做调和插值，保留摆动弧线的拉普拉斯坐标。"""
    targets = points.copy()
    mask = np.repeat(labels, 2, axis=2)
    for side in range(2):
        for corner in range(4):
            active = mask[:, side, corner]
            for a, b in runs(active):
                anchor = points[a:b, side, corner].mean(0)
                anchor[2] = 0.
                targets[a:b, side, corner] = anchor
            ids = np.flatnonzero(active)
            if len(ids):
                offsets = targets[ids, side, corner] - points[ids, side, corner]
                for axis in range(3):
                    targets[:, side, corner, axis] = points[:, side, corner, axis] + np.interp(
                        np.arange(len(points)), ids, offsets[:, axis])
    targets[..., 2] = np.maximum(targets[..., 2], 0.)
    return targets, mask


class MultiPointProblem(ContactWindowProblem):
    def __init__(self, poses, labels, config=MultiPointConfig(), fixed_frames=None):
        super().__init__(poses, labels.any(2), config, fixed_frames)
        self.targets, mask = target_trajectories(self.points, labels)
        self.mask = mask
        # 四点总权重归一化，避免仅因点数增加而将脚部损失扩大四倍。
        self.point_weights = np.where(mask, 1., config.swing_weight) / 4.

    def evaluate(self, x):
        if self.cached_x is not None and np.array_equal(x, self.cached_x):
            return self.cached
        c = self.cfg
        points, jac = self.geometry(x)
        w = np.sqrt(self.point_weights)
        scales = np.array([c.anchor_scale_m, c.anchor_scale_m, c.height_scale_m])
        anchor = (points - self.targets) * w[..., None] / scales
        aj = self.block(jac * w[..., None, None] / scales[None, None, None, :, None])
        penetration = np.minimum(points[..., 2], 0.) / (2 * c.penetration_scale_m)
        pj = self.block(jac[..., 2, :] * (points[..., 2] < 0)[..., None] / (2 * c.penetration_scale_m))
        pair = (self.mask[1:] & self.mask[:-1]).astype(float) / 4.
        weights = np.sqrt(pair) * c.fps / c.speed_scale_mps
        velocity = np.diff(points, axis=0) * weights[..., None]
        diff = sparse.diags([-np.ones(self.n-1), np.ones(self.n-1)], [0, 1], shape=(self.n-1, self.n))
        vj = sparse.diags(np.repeat(weights.ravel(), 3)) @ sparse.kron(diff, sparse.eye(24)) @ self.block(jac)
        residual = np.concatenate([self.reg @ x, anchor.ravel(), penetration.ravel(), velocity.ravel()])
        matrix = sparse.vstack([self.reg, aj, pj, vj], format='csr')
        self.cached_x = np.array(x, copy=True)
        self.cached = residual, matrix
        return self.cached

    def root_initialization(self):
        # 平移最小二乘的闭式解；弱摆动约束不驱动初始化。
        weights = self.mask.astype(float)
        shift = ((self.targets-self.points)*weights[..., None]).sum((1, 2)) / np.maximum(weights.sum((1, 2))[:, None], 1)
        x = np.zeros((self.n, 15))
        x[:, :3] = shift
        return np.clip(x, self.lower, self.upper).ravel()

    def smooth(self, x):
        # 平滑修正量；最后仍进行 IK，以免平滑重新破坏接触。
        diff = sparse.diags([np.ones(self.n-2), -2*np.ones(self.n-2), np.ones(self.n-2)], [0, 1, 2], shape=(self.n-2, self.n))
        matrix = sparse.eye(self.n, format='csc') + self.cfg.smoothing_strength * (diff.T @ diff)
        result = spsolve(matrix.tocsc(), x.reshape(self.n, 15))
        return np.clip(result, self.lower, self.upper).ravel()


def refine_contact_multipoint(poses, plan, split_labels=False, config=MultiPointConfig(), fixed_frames=None):
    start = time.perf_counter()
    base = np.asarray(poses, np.float32)
    plan = np.asarray(plan, bool)
    if base.ndim != 2 or base.shape[1] != 35 or len(base) < 3 or not np.isfinite(base).all():
        raise ValueError('需要至少三帧的有限 pose35 数组')
    if plan.shape != (len(base), 2):
        raise ValueError('Plan 应为 (T,2)')
    fk = FootGeometry()
    points, normals = fk.sequence(base)
    if split_labels:
        labels, eligible = region_labels(points, plan, config.fps)
    else:
        whole, _ = initial_labels(points, normals, plan, config.fps)
        labels = np.repeat(whole[..., None], 2, axis=2)
        eligible = None
    history = [labels.copy()]
    steps = []
    x = None
    for iteration in range(config.outer_iterations):
        problem = MultiPointProblem(base, labels, config, fixed_frames)
        if x is None:
            x = problem.root_initialization()
        else:
            x = problem.smooth(x)
        x, report = problem.solve(x)
        current = problem.pose(x)
        report['iteration'] = iteration
        report['active_region_frames'] = int(labels.sum())
        # A 固定每脚标签；B 按修复结果修正前后掌标签。最终标签确实参与最后一次求解。
        if split_labels and iteration+1 < config.outer_iterations:
            revised, _ = fk.sequence(current)
            labels, _ = region_labels(revised, plan, config.fps, eligible, labels)
        history.append(labels.copy())
        steps.append(report)
    current = current.astype(np.float32)
    current[:, 3:6] = base[:, 3:6]
    current[:, 18:] = base[:, 18:]
    current[problem.fixed | problem.airborne] = base[problem.fixed | problem.airborne]
    return current, np.asarray(history), dict(config=asdict(config), split_labels=split_labels,
        label_order=['heel', 'toe'], steps=steps, processing_seconds=time.perf_counter()-start,
        changed_region_labels=int((history[0] != history[-1]).sum()),
        root_rotation_optimized=False, upper_world_orientation_preserved=True,
        max_root_translation_cm=float(np.linalg.norm(current[:, :3]-base[:, :3], axis=1).max()),
        max_leg_change_deg=float(np.abs(current[:, 6:18]-base[:, 6:18]).max()))
