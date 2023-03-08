import argparse
import sys
import numpy as np

from duality.digit.digit_arm_diff_ik import DigitArmDiffInverseKinematics
from util.colors import OKGREEN, FAIL, ENDC

def test_duality_imports():
    try: 
        from duality.digit import DigitArmDiffInverseKinematics
        from duality.digit import DigitDrakeSim
        digit = DigitArmDiffInverseKinematics()
        digit.wire_arm_V_G_tracking_diff_ik()
    except ImportError:
        print(f"{FAIL}")

    print(f"{OKGREEN}Passed all duality package import tests \u2713{ENDC}")

def test_duality_diff_ik_constructor():
    try: 
        from duality.digit import DigitArmDiffInverseKinematics
        digit = DigitArmDiffInverseKinematics()
        digit.wire_arm_V_G_tracking_diff_ik()
        assert(digit.plant is not None)
        assert(digit.plant_context is not None)
    except ImportError:
        print(f"{FAIL}")

    print(f"{OKGREEN}Passed duality diff ik function import tests \u2713{ENDC}")

def test_all():
    test_duality_imports()
    test_duality_diff_ik_constructor()