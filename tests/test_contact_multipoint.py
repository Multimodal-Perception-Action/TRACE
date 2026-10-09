"""四点接触的数值雅可比、标签与轨迹插值测试。"""
import unittest
import numpy as np
from trace_core.foot_labels import FootGeometry
from trace_core.contact_multipoint import (MultiPointProblem, MultiPointConfig,
    target_trajectories, region_labels, refine_contact_multipoint)


def standing(n):
    pose = np.zeros((n,35),np.float32)
    points,_ = FootGeometry().sequence(pose)
    pose[:,1] -= points[...,2].min()*100
    return pose


class MultiPointTests(unittest.TestCase):
    def test_pipeline_exports(self):
        import tempfile
        import json
        from pathlib import Path
        from trace_core.pipeline import contact_refine
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);poses=standing(12)
            fixed=np.zeros(12,bool);fixed[:2]=True;fixed[-2:]=True
            np.save(out/'pose35_smoothed.npy',poses)
            np.save(out/'boundary_frame_mask.npy',fixed)
            np.savez(out/'plan.npz',frames=12,events=np.ones((3,16)),known=np.ones((3,16),bool))
            contact_refine({'contact_refine':'heel_toe','boundary':'standing'},out,'cpu')
            np.testing.assert_array_equal(np.load(out/'pose35_final.npy')[fixed],poses[fixed])
            self.assertEqual(np.load(out/'contact_labels_final.npy').shape,(12,2,2))
            self.assertTrue((out/'motion_final.sps').is_file())
            self.assertTrue((out/'motion_final_qpos.npz').is_file())
            self.assertEqual(json.loads((out/'contact_refine_report.json').read_text())['mode'],'heel_toe')

    def test_jacobian(self):
        p = MultiPointProblem(standing(5), np.ones((5,2,2),bool))
        rng = np.random.default_rng(7)
        x = rng.normal(size=75)*.001
        x.reshape(5,15)[:,2] -= .006
        d = rng.normal(size=75); d /= np.linalg.norm(d)
        r,j = p.evaluate(x)
        numeric = (p.evaluate(x+1e-7*d)[0]-r)/1e-7
        np.testing.assert_allclose(j@d,numeric,atol=1e-3,rtol=1e-3)

    def test_split_labels(self):
        points,_ = FootGeometry().sequence(standing(8))
        points[:,:,2:,2] += .12
        labels,_ = region_labels(points,np.ones((8,2),bool),20)
        self.assertTrue(labels[:,:,0].all())
        self.assertFalse(labels[:,:,1].any())

    def test_laplacian_and_ground(self):
        points = np.zeros((9,2,4,3))
        points[:,:,:,0] = np.arange(9)[:,None,None]*.01
        points[3:6,:,:,2] = .1
        labels = np.zeros((9,2,2),bool);labels[:3]=True;labels[6:]=True
        target,_ = target_trajectories(points,labels)
        self.assertGreater(target[4,0,0,2],.09)
        np.testing.assert_allclose(target[:3],np.broadcast_to(target[:1],target[:3].shape))
        self.assertGreaterEqual(target[...,2].min(),0)

    def test_preservation_and_bounds(self):
        poses = standing(9);poses[:,2]=np.linspace(0,3,9)
        fixed = np.zeros(9,bool);fixed[[0,-1]]=True
        for split in (False,True):
            out,hist,r = refine_contact_multipoint(poses,np.ones((9,2),bool),split,
                MultiPointConfig(outer_iterations=2,inner_evaluations=6),fixed)
            np.testing.assert_array_equal(out[fixed],poses[fixed])
            np.testing.assert_array_equal(out[:,3:6],poses[:,3:6])
            np.testing.assert_array_equal(out[:,18:],poses[:,18:])
            self.assertLessEqual(np.abs(out[:,:3]-poses[:,:3]).max(),5.0001)
            self.assertLessEqual(r['max_leg_change_deg'],12.0001)
            self.assertEqual(hist.shape,(3,9,2,2))

    def test_airborne(self):
        poses=standing(5);poses[:,1]+=40
        out,_,_=refine_contact_multipoint(poses,np.ones((5,2),bool),True,
            MultiPointConfig(outer_iterations=1,inner_evaluations=3))
        np.testing.assert_array_equal(out,poses)


if __name__=='__main__':unittest.main()
