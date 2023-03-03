import argparse
import sys
import numpy as np

from drail_drake.digit.digit_arm_diff_ik import DigitArmDiffInverseKinematics

def test_diff_ik():
    digit = DigitArmDiffInverseKinematics()
    digit.wire_arm_V_G_tracking_diff_ik()
    v_gw_l = np.ones(6)
    v_gw_r = -np.ones(6)
    q_l = np.zeros(4)
    left, right = digit.do_arm_diff_ik(q_l,
                         q_l,
                         q_l,
                         q_l,
                         v_gw_l,
                         v_gw_r)
    print(left)
    print(right)
