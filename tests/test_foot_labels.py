"""脚步实验的数学与保护性约束测试，不创建 OpenGL 上下文。"""
import unittest
import numpy as np
from trace_core.foot_labels import FootGeometry, clean_mask, label_ik, rule_support


class FootTests(unittest.TestCase):
    def test_short_contact_removed(self):
        mask=np.zeros((8,2),bool);mask[:2,0]=True;mask[3:7,1]=True
        cleaned=clean_mask(mask)
        self.assertFalse(cleaned[:,0].any());self.assertEqual(cleaned[:,1].sum(),4)

    def test_no_labels_no_change(self):
        pose=np.zeros((8,35),np.float32);pose[:,1]=80
        result,_,_=label_ik(pose,np.zeros((8,2),bool))
        np.testing.assert_array_equal(pose,result)

    def test_jacobian_matches_finite_difference(self):
        fk=FootGeometry();pose=np.zeros(35);pose[1]=80;pose[9]=25;fk.set_pose(pose)
        analytic=fk.jacobian(0);origin=fk.sole(0).mean(0)
        numeric=[]
        for adr in fk.qa[:6]:
            old=fk.data.qpos[adr];fk.data.qpos[adr]+=1e-6;fk.mj.mj_forward(fk.model,fk.data)
            numeric.append((fk.sole(0).mean(0)-origin)/1e-6)
            fk.data.qpos[adr]=old;fk.mj.mj_forward(fk.model,fk.data)
        np.testing.assert_allclose(analytic,np.array(numeric).T,atol=1e-6)

    def test_airborne_plan_not_locked(self):
        pose=np.zeros((8,35),np.float32);pose[:,1]=180
        result,mask,_=label_ik(pose,np.ones((8,2),bool),guarded=True)
        self.assertFalse(mask.any());np.testing.assert_array_equal(pose,result)

    def test_ik_protects_other_coordinates(self):
        pose=np.zeros((12,35),np.float32);pose[:,1]=80;pose[:,2]=np.linspace(0,5,12)
        result,_,report=label_ik(pose,np.ones((12,2),bool),guarded=False)
        self.assertGreater(report['max_leg_change_deg'],0)
        self.assertLessEqual(report['max_leg_change_deg'],12.0001)
        np.testing.assert_array_equal(pose[:,:6],result[:,:6])
        np.testing.assert_array_equal(pose[:,18:],result[:,18:])


if __name__=='__main__':unittest.main()
