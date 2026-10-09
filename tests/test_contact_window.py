"""联合窗口求解的 Jacobian、时间耦合、root 参与和标签更新测试。"""
import unittest
import numpy as np
from trace_core.foot_labels import FootGeometry
from trace_core.contact_window import ContactWindowProblem,WindowConfig,refine_contact_window,revise_labels


def standing(n=8):
    poses=np.zeros((n,35),np.float32);poses[:,1]=80
    points,_=FootGeometry().sequence(poses)
    poses[:,1]-=points[...,2].min()*100
    return poses


class ContactWindowTests(unittest.TestCase):
    def test_fixed_boundary_in_solver(self):
        poses=standing(10);poses[:,2]=np.linspace(0,4,10)
        fixed=np.zeros(10,bool);fixed[:2]=True;fixed[-2:]=True
        p=ContactWindowProblem(poses,np.ones((10,2),bool),fixed_frames=fixed)
        np.testing.assert_array_equal(p.pose(np.ones(150)*.001)[fixed],poses[fixed])
        _,jac=p.geometry(np.zeros(150))
        self.assertEqual(np.abs(jac[fixed]).max(),0.)
        out,_,report=refine_contact_window(poses,np.ones((10,2),bool),True,
            WindowConfig(outer_iterations=2,inner_evaluations=5),fixed_frames=fixed)
        np.testing.assert_array_equal(out[fixed],poses[fixed])
        self.assertEqual(report['fixed_frames'],4)

    def test_analytic_directional_derivative(self):
        poses=standing(5);problem=ContactWindowProblem(poses,np.ones((5,2),bool))
        rng=np.random.default_rng(123);x=rng.normal(size=75)*.001;x.reshape(5,15)[:,2]-=.005
        direction=rng.normal(size=75);direction/=np.linalg.norm(direction)
        residual,jac=problem.evaluate(x)
        numeric=(problem.evaluate(x+direction*1e-7)[0]-residual)/1e-7
        np.testing.assert_allclose(jac@direction,numeric,atol=1e-3,rtol=1e-3)

    def test_temporal_coupling_is_nonzero(self):
        p=ContactWindowProblem(standing(5),np.ones((5,2),bool))
        gram=(p.reg.T@p.reg).toarray()
        self.assertGreater(np.abs(gram[:15,15:30]).sum(),0)

    def test_root_participates_and_limits_hold(self):
        poses=standing(10);poses[:,2]=np.linspace(0,4,10)
        out,history,report=refine_contact_window(poses,np.ones((10,2),bool),False,
            WindowConfig(outer_iterations=1,inner_evaluations=15))
        self.assertGreater(report['max_root_translation_cm'],.05)
        self.assertLessEqual(np.abs(out[:,:3]-poses[:,:3]).max(),5.0001)
        self.assertLessEqual(report['max_leg_change_deg'],12.0001)
        self.assertLess(report['steps'][0]['cost_after'],report['steps'][0]['cost_before'])
        np.testing.assert_array_equal(out[:,3:6],poses[:,3:6])
        np.testing.assert_array_equal(out[:,18:],poses[:,18:])
        np.testing.assert_array_equal(history[0],history[-1])

    def test_airborne_frames_not_changed(self):
        poses=standing(5);poses[:,1]+=40
        out,_,report=refine_contact_window(poses,np.ones((5,2),bool),True,
            WindowConfig(outer_iterations=1,inner_evaluations=3))
        np.testing.assert_array_equal(out,poses)
        self.assertEqual(report['frozen_airborne_frames'],5)

    def test_label_revision_adds_and_removes(self):
        points,_=FootGeometry().sequence(standing(8))
        old=np.zeros((8,2),bool);old[:,1]=True
        points[:,1,:,2]+=.2
        updated=revise_labels(points,old,np.zeros_like(old),np.ones_like(old))
        self.assertTrue(updated[:,0].all());self.assertFalse(updated[:,1].any())


if __name__=='__main__':unittest.main()
