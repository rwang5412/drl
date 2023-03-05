import argparse
import sys
import numpy as np

from drail_drake.digit.digit_arm_diff_ik import DigitArmDiffInverseKinematics
from util.colors import OKGREEN, FAIL, ENDC

def test_drail_drake_imports():
    try: 
        from drail_drake.digit import DigitArmDiffInverseKinematics
        from drail_drake.digit import DigitDrakeSim
    except ImportError:
        print(f"{FAIL}")
    
    print(f"{OKGREEN}Passed all drail drake package import tests \u2713{ENDC}")
