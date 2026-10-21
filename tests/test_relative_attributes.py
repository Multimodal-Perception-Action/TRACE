"""Signed goals, kinematic scopes and independent MuJoCo geometry checks."""
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from trace_core.conversation import compile_intent, parse_intent, summarize_motion, validate_explicit_distance
from trace_core.editor_geometry import G1SoleForwardKinematics
from trace_core.motion_io import to_gravity_gauge
from trace_core.relative_attribute_guidance import relative_peak_target, relative_attribute_mask, RelativeAttributeGuidance, validate_attribute_report


class RelativeAttributeTests(unittest.TestCase):
    def test_target_direction_and_current_source(self):
        x = np.full(70, 35.)
        x[8:62] += 20*np.sin(np.linspace(0, np.pi, 54))**2
        for delta in [-5, 5]:
            t, w, r = relative_peak_target(x, [8,62], delta, 'knee_height')
            self.assertAlmostEqual(float(t.max()-x.max()), delta, places=4)
            np.testing.assert_array_equal(t[:8], x[:8])
            np.testing.assert_array_equal(t[-8:], x[-8:])
            t2, _, _ = relative_peak_target(t, [8,62], delta, 'knee_height')
            self.assertAlmostEqual(float(t2.max()-x.max()), 2*delta, places=4)
        with self.assertRaisesRegex(ValueError, 'No measurable'):
            relative_peak_target(np.full(70,35), [8,62], -5, 'knee_height')
        with self.assertRaises(ValueError):
            relative_peak_target(x, [8,62], -25, 'knee_height')

    def test_intent_units_and_signs(self):
        x = np.full(70, 0.)
        x[8:62] = 30*np.sin(np.linspace(0, np.pi,54))**2
        summary = dict(frames=70,boundary_frames=8,attribute_profiles={'bow_angle':x.tolist()})
        for direction, sign in [('deeper',1),('shallower',-1)]:
            value = dict(action='bow_angle',target_caption='Bow with a different angle.',body='torso',direction=direction,
                         magnitude='moderate',amount_cm=None,amount_deg=None,time_region='full',reason='Relative bow angle')
            value = parse_intent(json.dumps(value))
            self.assertEqual(compile_intent(value,summary)['attribute_delta'], sign*5)
            value['amount_deg'] = 4
            validate_explicit_distance(value,'Bow four degrees deeper.')
            with self.assertRaises(ValueError):
                validate_explicit_distance(value,'Bow more deeply.')
            value['amount_cm'] = 5
            with self.assertRaises(ValueError):
                parse_intent(json.dumps(value))

    def test_fk_matches_independent_mujoco(self):
        from trace_core.foot_labels import FootGeometry
        fk = FootGeometry()
        x = np.zeros((35,35),np.float32)
        x[:,1]=80
        x[:,6]=-25
        x[:,9]=60
        x[:,12]=-10
        x[:,15]=30
        x[:,3:6]=[2,5,0]
        x[:,18:21]=[6,-4,12]
        raw = torch.from_numpy(to_gravity_gauge(x))[None]
        lm = G1SoleForwardKinematics().landmarks(raw,torch.ones((1,len(x)),dtype=torch.bool))
        fk.set_pose(x[0])
        for side,name in enumerate(('left_knee_link','right_knee_link')):
            self.assertAlmostEqual(float(lm['knees'][0,0,side,2]),fk.data.xpos[fk.model.body(name).id,2],places=5)
        up=fk.data.xmat[fk.model.body('torso_link').id].reshape(3,3)[:,2]
        self.assertAlmostEqual(float(lm['torso_pitch_deg'][0,0]),float(np.rad2deg(np.arctan2(up[0],up[2]))),places=4)

    def test_scope_and_projection(self):
        n=70
        for kind,side in [('kick_height',1),('knee_height',0),('bow_angle',0)]:
            mask=relative_attribute_mask(n,[8,62],kind,side)
            self.assertFalse(mask[:8].any())
            self.assertFalse(mask[-8:].any())
            self.assertFalse(mask[:,23:].any())
            self.assertFalse(mask[:,:5].any())
            if kind!='bow_angle':
                self.assertFalse(mask[:,5:11].any())
                self.assertEqual(int(mask.sum()),54*6)

    def test_failure_guard(self):
        report=dict(achieved_peak=25,target_peak=25,source_peak=20,requested_change=5,
                    profile_rmse=.2,max_added_penetration_cm=.1,attribute='kick_height')
        validate_attribute_report(report)
        for changes in [dict(achieved_peak=19),dict(profile_rmse=3),dict(max_added_penetration_cm=2)]:
            with self.assertRaises(ValueError):validate_attribute_report(dict(report,**changes))

    def test_bow_projection_removes_hidden_heading(self):
        from trace_core.editor_geometry import rot6d_to_matrix_torch
        x=np.zeros((70,35),np.float32)
        x[:,1]=80
        x[8:62,4]=30*np.sin(np.linspace(0,np.pi,54))**2
        source=torch.from_numpy(to_gravity_gauge(x))[None]
        mask=torch.from_numpy(relative_attribute_mask(70,[8,62],'bow_angle',0))[None]
        guide=RelativeAttributeGuidance(source,np.zeros(40),np.ones(40),mask,[8,62],'bow_angle',0,5)
        dirty=source.clone()
        dirty[...,6]+=.15
        output=guide.project(dirty)
        rotation=rot6d_to_matrix_torch(output[...,5:11])
        yaw=torch.atan2(rotation[...,1,0],rotation[...,0,0])
        self.assertLess(float(yaw.abs().max()),1e-6)
        self.assertTrue(torch.equal(output[~mask],source[~mask]))


if __name__=='__main__': unittest.main()
