"""离线多帧接触修复：root 平移与双腿联合优化，交替更新接触标签。"""
from dataclasses import asdict, dataclass
import time
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from .foot_labels import FootGeometry, clean_mask, runs


@dataclass(frozen=True)
class WindowConfig:
    fps: int = 20
    outer_iterations: int = 3
    inner_evaluations: int = 35
    root_limit_m: float = .05
    leg_limit_deg: float = 12.
    anchor_scale_m: float = .018
    speed_scale_mps: float = .12
    height_scale_m: float = .012
    penetration_scale_m: float = .004


def contact_signals(points, fps):
    centers=points.mean(2)
    velocity=np.zeros_like(centers)
    if len(points)>1:
        velocity[1:]=np.diff(centers,axis=0)*fps
        velocity[0]=velocity[1]
    return points[...,2].min(2),np.linalg.norm(velocity[...,:2],axis=-1),np.abs(velocity[...,2])


def initial_labels(points, normals, plan, fps=20):
    """语言标签为先验；低位慢速脚可以补标签，原始腾空证据阻止误锁。"""
    h,v,vz=contact_signals(points,fps)
    eligible=(h<=.065)&(normals>=np.cos(np.deg2rad(50)))&(vz<.6)
    labels=eligible&(plan|((h<=.03)&(v<=.15)&(vz<=.20)))
    return clean_mask(labels),eligible


def revise_labels(points, previous, plan, eligible, fps=20):
    """不同进入/退出阈值形成滞回；允许补漏标和移除冲突标记。"""
    h,v,vz=contact_signals(points,fps)
    keep=previous&(h<.065)&(v<.55)&(vz<.45)
    enter=(h<.025)&(v<.13)&(vz<.18)
    enter|=plan&(h<.035)&(v<.22)&(vz<.25)
    labels=(keep|enter)&eligible
    # 仅填补单帧内部空洞，且必须仍满足原动作的接触候选门槛。
    if len(labels)>2:labels[1:-1]|=labels[:-2]&labels[2:]&eligible[1:-1]
    return clean_mask(labels)


def contact_weights(mask):
    weights=mask.astype(float)
    for side in range(2):
        for a,b in runs(mask[:,side]):
            fade=min(3,(b-a)//2)
            ramp=.5-.5*np.cos(np.linspace(0,np.pi,fade+2)[1:-1])
            weights[a:a+fade,side]*=ramp
            weights[b-fade:b,side]*=ramp[::-1]
    return weights


class ContactWindowProblem:
    """整个短序列作为联合窗口；解析 FK Jacobian 与稀疏时间差分共同求解。"""
    def __init__(self, poses, labels, config=WindowConfig(), fixed_frames=None):
        self.base=np.asarray(poses,dtype=float);self.n=len(poses);self.cfg=config
        self.fixed=np.zeros(self.n,bool) if fixed_frames is None else np.asarray(fixed_frames,bool)
        if self.fixed.shape!=(self.n,):raise ValueError('固定帧 mask 尺寸应为 (T,)')
        self.fk=FootGeometry();self.weights=contact_weights(labels)
        points,_=self.fk.sequence(self.base);self.points=points
        self.targets=points.mean(2)[...,:2].copy()
        for side in range(2):
            for a,b in runs(labels[:,side]):
                # 每一轮均从未经 IK 的输入确定锚点，避免锚点随修复结果漂移。
                self.targets[a:b,side]=np.median(points[a:min(a+3,b),side].mean(1)[:,:2],axis=0)
        def difference(order):
            if self.n<=order:return sparse.csr_matrix((0,self.n))
            coefficients=([-1.,1.] if order==1 else [1.,-2.,1.])
            return sparse.diags([np.full(self.n-order,c) for c in coefficients],
                list(range(order+1)),shape=(self.n-order,self.n),format='csr')
        # 正则项作用于修正量，优先保持已经平滑的输入运动。
        keep=np.r_[np.full(3,1/.035),np.full(12,1/.20)]
        vel=np.r_[np.full(3,1/.012),np.full(12,1/.10)]
        acc=np.r_[np.full(3,1/.004),np.full(12,1/.025)]
        self.reg=sparse.vstack([sparse.kron(sparse.eye(self.n),sparse.diags(keep)),
            sparse.kron(difference(1),sparse.diags(vel)),
            sparse.kron(difference(2),sparse.diags(acc))],format='csr')
        cap=np.r_[np.full(3,config.root_limit_m),np.full(12,np.deg2rad(config.leg_limit_deg))]
        self.lower=np.tile(-cap,(self.n,1));self.upper=np.tile(cap,(self.n,1))
        angles=np.deg2rad(self.base[:,6:18]);limits=self.fk.limits[:12]
        valid=(angles>=limits[:,0])&(angles<=limits[:,1]);self.invalid_count=int((~valid).sum())
        self.lower[:,3:]=np.maximum(self.lower[:,3:],limits[:,0]-angles)
        self.upper[:,3:]=np.minimum(self.upper[:,3:],limits[:,1]-angles)
        # 输入已越限的自由度保持原值，不用裁剪掩盖输入问题。
        self.lower[:,3:]=np.where(valid,self.lower[:,3:],-1e-12)
        self.upper[:,3:]=np.where(valid,self.upper[:,3:],1e-12)
        self.lower=np.minimum(self.lower,-1e-12);self.upper=np.maximum(self.upper,1e-12)
        self.airborne=(points[...,2].min(2)>.08).all(1)
        self.lower[self.airborne]=-1e-12;self.upper[self.airborne]=1e-12
        self.lower[self.fixed]=-1e-12;self.upper[self.fixed]=1e-12
        self.cached_x=None

    def pose(self,x):
        delta=np.asarray(x).reshape(self.n,15);pose=self.base.copy()
        pose[:,[2,0,1]]+=delta[:,:3]*100
        pose[:,6:18]+=np.rad2deg(delta[:,3:])
        # 固定帧直接参与邻帧残差，但自身姿态始终不变，并非求解后才覆盖。
        pose[self.fixed]=self.base[self.fixed]
        return pose

    def geometry(self,x):
        pose=self.pose(x);points=[];jacobians=[]
        for t in range(self.n):
            self.fk.set_pose(pose[t]);pts=np.stack([self.fk.sole(s) for s in range(2)])
            jac=np.zeros((2,4,3,15));jac[...,:3]=np.eye(3)
            for side in range(2):
                for corner in range(4):
                    full=np.zeros((3,self.fk.model.nv))
                    self.fk.mj.mj_jac(self.fk.model,self.fk.data,full,None,pts[side,corner],self.fk.body[side])
                    jac[side,corner,:,3:]=full[:,self.fk.va[:12]]
            if self.fixed[t]:jac[:]=0
            points.append(pts);jacobians.append(jac)
        return np.asarray(points),np.asarray(jacobians)

    def block(self,jac):
        """将每帧的局部 Jacobian 放到全序列稀疏对角块。"""
        jac=jac.reshape(self.n,-1,15);width=jac.shape[1]
        rows=np.broadcast_to(np.arange(self.n*width).reshape(self.n,width,1),jac.shape)
        cols=np.broadcast_to(np.arange(self.n*15).reshape(self.n,1,15),jac.shape)
        return sparse.coo_matrix((jac.ravel(),(rows.ravel(),cols.ravel())),shape=(self.n*width,self.n*15)).tocsr()

    def evaluate(self,x):
        if self.cached_x is not None and np.array_equal(x,self.cached_x):return self.cached
        c=self.cfg;points,jac=self.geometry(x);centers=points.mean(2);cj=jac.mean(2)
        w=np.sqrt(self.weights)
        anchor=(centers[...,:2]-self.targets)*w[...,None]/c.anchor_scale_m
        aj=self.block(cj[...,:2,:]*w[...,None,None]/c.anchor_scale_m)
        low=np.argmin(points[...,2],axis=2);ti=np.arange(self.n)[:,None];si=np.arange(2)[None,:]
        bottom=points[ti,si,low,2];bj=jac[ti,si,low,2,:]
        height=bottom*w/c.height_scale_m
        hj=self.block(bj*w[...,None]/c.height_scale_m)
        penetrating=points[...,2]<0
        penetration=np.minimum(points[...,2],0)/c.penetration_scale_m
        pj=self.block(jac[...,2,:]*penetrating[...,None]/c.penetration_scale_m)
        pair=np.sqrt(self.weights[1:]*self.weights[:-1])
        velocity=np.diff(centers[...,:2],axis=0)*pair[...,None]*c.fps/c.speed_scale_mps
        # 足部速度残差同时依赖前后帧，显式连接相邻帧变量。
        centers_matrix=self.block(cj[...,:2,:])
        diff=sparse.diags([-np.ones(max(0,self.n-1)),np.ones(max(0,self.n-1))],[0,1],shape=(max(0,self.n-1),self.n))
        vj=sparse.diags(np.repeat(pair.ravel(),2)*c.fps/c.speed_scale_mps)@sparse.kron(diff,sparse.eye(4))@centers_matrix
        residual=np.concatenate([self.reg@x,anchor.ravel(),height.ravel(),penetration.ravel(),velocity.ravel()])
        matrix=sparse.vstack([self.reg,aj,hj,pj,vj],format='csr')
        self.cached_x=np.array(x,copy=True);self.cached=(residual,matrix)
        return self.cached

    def solve(self,initial):
        initial=np.clip(initial,self.lower.ravel()+1e-14,self.upper.ravel()-1e-14)
        before=float(np.sum(self.evaluate(initial)[0]**2))
        fit=least_squares(lambda x:self.evaluate(x)[0],initial,jac=lambda x:self.evaluate(x)[1],
            bounds=(self.lower.ravel(),self.upper.ravel()),method='trf',tr_solver='lsmr',
            x_scale=np.tile(np.r_[np.full(3,.025),np.full(12,.1)],self.n),
            max_nfev=self.cfg.inner_evaluations,ftol=1e-5,xtol=1e-5,gtol=1e-5)
        return fit.x,dict(cost_before=before,cost_after=float(2*fit.cost),evaluations=int(fit.nfev),
            optimizer_status=int(fit.status),optimizer_message=fit.message)


def refine_contact_window(poses,plan,update_labels=True,config=WindowConfig(),fixed_frames=None):
    """后处理接口：不运行或训练大模型，不读取 GT。"""
    start=time.perf_counter();base=np.asarray(poses,dtype=np.float32)
    if base.ndim!=2 or base.shape[1]!=35 or len(base)<3 or not np.isfinite(base).all():
        raise ValueError('需要至少三帧的有限 pose35 数组')
    plan=np.asarray(plan,dtype=bool)
    if plan.shape!=(len(base),2):raise ValueError('Plan 标签尺寸应为 (T,2)')
    fk=FootGeometry();points,normals=fk.sequence(base)
    labels,eligible=initial_labels(points,normals,plan,config.fps);initial=labels.copy()
    history=[labels.copy()];steps=[];x=np.zeros(len(base)*15);current=base.copy()
    # 固定和可更新标签版本使用相同外层次数及求解预算。
    for outer in range(config.outer_iterations):
        problem=ContactWindowProblem(base,labels,config,fixed_frames)
        x,report=problem.solve(x);current=problem.pose(x)
        report.update(iteration=outer,contact_frames=int(labels.sum()),initial_outside_limits=problem.invalid_count)
        if update_labels and outer+1<config.outer_iterations:
            revised_points,_=fk.sequence(current)
            revised=revise_labels(revised_points,labels,plan,eligible,config.fps)
            report.update(added_labels=int((revised&~labels).sum()),removed_labels=int((labels&~revised).sum()))
            labels=revised
        else:report.update(added_labels=0,removed_labels=0)
        history.append(labels.copy());steps.append(report)
    current=current.astype(np.float32)
    # 浮点转换后明确恢复不在优化变量中的坐标。
    current[:,3:6]=base[:,3:6];current[:,18:]=base[:,18:]
    current[problem.airborne]=base[problem.airborne]
    current[problem.fixed]=base[problem.fixed]
    report=dict(config=asdict(config),update_labels=update_labels,root_translation_optimized=True,
        root_rotation_optimized=False,upper_joint_angles_preserved=True,
        upper_world_positions_preserved=False,whole_clip_joint_window=True,
        initial_labels=int(initial.sum()),final_labels=int(labels.sum()),
        changed_labels=int((initial!=labels).sum()),steps=steps,processing_seconds=time.perf_counter()-start,
        max_root_translation_cm=float(np.linalg.norm(current[:,:3]-base[:,:3],axis=1).max()),
        max_leg_change_deg=float(np.abs(current[:,6:18]-base[:,6:18]).max()),
        frozen_airborne_frames=int(problem.airborne.sum()),fixed_frames=int(problem.fixed.sum()))
    return current,np.asarray(history),report
