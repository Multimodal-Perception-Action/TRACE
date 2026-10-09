"""脚步标签与锁脚实验；不自动接入正式生成入口。所有几何量使用米、弧度。"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .motion_io import SPS_JOINTS

ROOT = Path(__file__).resolve().parents[1]
SOLE = np.array([[-.05,.025,-.035],[-.05,-.025,-.035],[.12,.03,-.035],[.12,-.03,-.035]])


def runs(mask, minimum=1):
    edges = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, bool), False]))
    return [(int(a), int(b)) for a,b in edges.reshape(-1,2) if b-a >= minimum]


def clean_mask(mask, minimum=3):
    result = np.zeros_like(mask, dtype=bool)
    for side in range(2):
        for a,b in runs(mask[:,side], minimum):
            result[a:b,side] = True
    return result


def plan_support(path, frames):
    """LLM 的已知支撑类别不是校准概率；未知标签不参与锁脚。"""
    with np.load(path, allow_pickle=False) as f:
        if int(f['frames']) != frames:
            raise ValueError('Plan 与动作帧数不一致')
        values = np.repeat(f['events'][:,:2], 4, axis=0)[:frames]
        known = np.repeat(f['known'][:,:2], 4, axis=0)[:frames]
    return (values >= .75) & known, known


class FootGeometry:
    """直接使用随包 G1 模型和解析雅可比，避免手工骨骼与可视化不一致。"""
    def __init__(self):
        import mujoco as mj
        self.mj = mj
        self.model = mj.MjModel.from_xml_path(str(ROOT/'assets/unitree_g1/g1_mocap_29dof.xml'))
        self.data = mj.MjData(self.model)
        ids = np.array([mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_JOINT,n) for n in SPS_JOINTS[1:]])
        self.qa = self.model.jnt_qposadr[ids]
        self.va = self.model.jnt_dofadr[ids]
        self.limits = self.model.jnt_range[ids]
        self.body = [mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_BODY,n) for n in ('left_ankle_roll_link','right_ankle_roll_link')]

    def set_pose(self, pose):
        self.data.qpos[:3] = pose[[2,0,1]]*.01
        self.data.qpos[3:7] = Rotation.from_euler('xyz',pose[3:6],degrees=True).as_quat(scalar_first=True)
        self.data.qpos[self.qa] = np.deg2rad(pose[6:])
        self.data.qvel[:] = 0
        self.mj.mj_forward(self.model,self.data)

    def sole(self, side):
        b = self.body[side]
        return SOLE @ self.data.xmat[b].reshape(3,3).T + self.data.xpos[b]

    def sequence(self, poses):
        points, normal_z = [], []
        for pose in poses:
            self.set_pose(pose)
            points.append(np.stack([self.sole(s) for s in range(2)]))
            normal_z.append([self.data.xmat[b].reshape(3,3)[2,2] for b in self.body])
        return np.array(points), np.array(normal_z)

    def jacobian(self, side):
        jac = np.zeros((3,self.model.nv))
        self.mj.mj_jac(self.model,self.data,jac,None,self.sole(side).mean(0),self.body[side])
        return jac[:, self.va[6*side:6*side+6]]


def rule_support(points, fps=20):
    """推理规则使用固定地面 z=0；不读取 GT，不逐方法重估地面。"""
    speed = np.zeros(points.shape[:2])
    if len(points)>1:
        speed[1:] = np.linalg.norm(np.diff(points.mean(2)[...,:2],axis=0),axis=-1)*fps
        speed[0] = speed[1]
    return clean_mask((points[...,2].min(2) <= .035) & (speed <= .20))


def label_ik(poses, requested, guarded=True):
    """仅修改双腿；对比硬锁脚与含可行性检查的渐进锁脚。"""
    poses = np.asarray(poses, dtype=np.float32)
    if requested.shape != (len(poses),2) or not np.isfinite(poses).all():
        raise ValueError('输入动作或标签无效')
    fk = FootGeometry()
    points, normal_z = fk.sequence(poses)
    mask = np.asarray(requested,bool).copy()
    if guarded:
        # 高抬脚或脚面大幅倾斜时，不把语言计划硬解释成实际接地。
        mask &= (points[...,2].min(2) <= .08) & (normal_z >= np.cos(np.deg2rad(40)))
        mask = clean_mask(mask)
    weights = mask.astype(float)
    if guarded:
        for side in range(2):
            for a,b in runs(mask[:,side]):
                fade = min(3,(b-a)//2)
                ramp = .5-.5*np.cos(np.linspace(0,np.pi,fade+2)[1:-1])
                weights[a:a+fade,side] *= ramp
                weights[b-fade:b,side] *= ramp[::-1]
    result = poses.copy()
    rejected, outside_limits = 0, 0
    for side in range(2):
        leg = slice(6+side*6,12+side*6)
        adr = fk.qa[side*6:side*6+6]
        for a,b in runs(mask[:,side]):
            # 锚点使用接触段原始前三帧脚底中心的中位数，保留原有高度。
            anchor = np.median(points[a:min(a+3,b),side].mean(1),axis=0)
            for t in range(a,b):
                fk.set_pose(result[t])
                initial = np.deg2rad(poses[t,leg]).copy()
                limits = fk.limits[side*6:side*6+6]
                if np.any(initial<limits[:,0]-1e-7) or np.any(initial>limits[:,1]+1e-7):
                    outside_limits += 1
                    continue
                low = np.maximum(limits[:,0],initial-np.deg2rad(12))
                high = np.minimum(limits[:,1],initial+np.deg2rad(12))
                original = points[t,side].mean(0)
                target = original+weights[t,side]*(anchor-original)
                base_depth = max(0.,-points[t,side,:,2].min())
                for _ in range(8):
                    current = fk.sole(side).mean(0)
                    error = target-current
                    if np.linalg.norm(error)<1e-4:
                        break
                    jac = fk.jacobian(side)
                    delta = jac.T @ np.linalg.solve(jac@jac.T + .05**2*np.eye(3), error)
                    delta = np.clip(delta,-np.deg2rad(2),np.deg2rad(2))
                    old = fk.data.qpos[adr].copy()
                    accepted = False
                    for scale in ([1,.5,.25,.125] if guarded else [1]):
                        fk.data.qpos[adr] = np.clip(old+scale*delta,low,high)
                        fk.mj.mj_forward(fk.model,fk.data)
                        new_points = fk.sole(side)
                        good = np.linalg.norm(target-new_points.mean(0)) <= np.linalg.norm(error)+1e-9
                        if guarded:
                            good &= max(0.,-new_points[:,2].min()) <= base_depth+.0005
                        if good:
                            accepted = True
                            break
                    if not accepted:
                        fk.data.qpos[adr] = old
                        fk.mj.mj_forward(fk.model,fk.data)
                        rejected += 1
                        break
                result[t,leg] = np.rad2deg(fk.data.qpos[adr])
    assert np.array_equal(result[:,:6],poses[:,:6]) and np.array_equal(result[:,18:],poses[:,18:])
    return result, mask, dict(guarded=guarded,requested_frames=int(requested.sum()),
        accepted_label_frames=int(mask.sum()),rejected_solver_steps=rejected,initial_outside_limits=outside_limits,
        max_leg_change_deg=float(np.max(np.abs(result[:,6:18]-poses[:,6:18]))),root_unchanged=True,upper_body_unchanged=True)


def diagnostics(poses, base, fixed_plan, fixed_rule):
    """所有方法共用预处理前冻结的评价标签，防止按各自输出重新选低速帧。"""
    points,_ = FootGeometry().sequence(poses)
    centers = points.mean(2)
    speed = np.linalg.norm(np.diff(centers[...,:2],axis=0)*20,axis=-1)
    row = {}
    for name,mask in [('plan',fixed_plan),('rule',fixed_rule)]:
        pair = mask[1:] & mask[:-1]
        row[f'{name}_pair_count'] = int(pair.sum())
        row[f'{name}_speed_mps'] = float(speed[pair].mean()) if pair.any() else None
        drifts=[]
        for side in range(2):
            for a,b in runs(mask[:,side],3):
                drifts.append(float(np.linalg.norm(centers[a:b,side,:2]-centers[a,side,:2],axis=-1).mean()*100))
        row[f'{name}_drift_cm'] = float(np.mean(drifts)) if drifts else None
    row.update(penetration_mean_cm=float(np.maximum(-points[...,2],0).mean()*100),
        penetration_max_cm=float(np.maximum(-points[...,2],0).max()*100),
        leg_acceleration_rms_deg_s2=float(np.sqrt(np.mean((np.diff(poses[:,6:18],n=2,axis=0)*400)**2))) if len(poses)>2 else None,
        leg_change_rms_deg=float(np.sqrt(np.mean((poses[:,6:18]-base[:,6:18])**2))),
        root_change_max_cm=float(np.max(np.abs(poses[:,:3]-base[:,:3]))),
        upper_change_max_deg=float(np.max(np.abs(poses[:,18:]-base[:,18:]))))
    return row
