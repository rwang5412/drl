import torch
import numpy as np
from env.cassie.cassieenv import CassieEnv
from env.digit.digitenv import DigitEnv
from util.mirror import mirror_tensor

"""This file tests if the mirror inds (from hand written one) are correct when creating a new set
of mirror inds for Digit/Cassie on robot proprioceptive state.
The test is meant for base env classes, with get_robot_state() and robot_state_mirror_indices.
The test mirrors the state once, then check if the mirrored states are correct.
"""

def test_mirror():
    cassie = CassieEnv(simulator_type='mujoco',
                    terrain=False,
                    policy_rate=50,
                    dynamics_randomization=True)

    digit = DigitEnv(simulator_type='mujoco',
                    terrain=False,
                    policy_rate=50,
                    dynamics_randomization=True)

    # Check if any mirror ind array has duplicate values by mistake
    assert len(np.unique(np.abs(cassie.robot_state_mirror_indices))) == len(cassie.get_robot_state()), \
        f"Cassie state mirror inds have duplicate values {np.sort(np.abs(cassie.robot_state_mirror_indices))}."
    assert len(np.unique(np.abs(cassie.motor_mirror_indices))) == cassie.sim.num_actuators, \
        f"Cassie motor mirror inds have duplicate values {np.sort(np.abs(cassie.motor_mirror_indices))}."

    assert len(np.unique(np.abs(digit.robot_state_mirror_indices))) == len(digit.get_robot_state()), \
        f"Digit state mirror inds have duplicate values {np.sort(np.abs(digit.robot_state_mirror_indices))}."
    assert len(np.unique(np.abs(digit.motor_mirror_indices))) == digit.sim.num_actuators, \
        f"Digit motor mirror inds have duplicate values {np.sort(np.abs(digit.motor_mirror_indices))}."

    cassie.reset_simulation()
    digit.reset_simulation()

    cassie_state = cassie.get_robot_state()
    digit_state = digit.get_robot_state()

    cassie_state_mirror_indices = [0.01, -1, 2, -3,      # base orientation
                                    -4, 5, -6,             # base rotational vel
                                    -12, -13, 14, 15, 16,  # right motor pos
                                    -7,  -8,  9,  10,  11, # left motor pos
                                    -22, -23, 24, 25, 26,  # right motor vel
                                    -17, -18, 19, 20, 21,  # left motor vel
                                    29, 30, 27, 28,        # joint pos
                                    33, 34, 31, 32, ]      # joint vel
    cassie_mirrored_left_motor_pos = mirror_tensor(torch.tensor(cassie_state), [-7,  -8,  9,  10,  11])
    cassie_right_motor_pos = np.take(cassie_state, [12, 13, 14, 15, 16])
    assert np.linalg.norm(cassie_mirrored_left_motor_pos - cassie_right_motor_pos) < 1e-6, \
        "Mirror incorrect"

    digit_state_mirror_indices = [0.01, -1, 2, -3,            # base orientation
                                -4, 5, -6,                    # base rotational vel
                                -17, -18, -19, -20, -21, -22, # right leg motor pos
                                -23, -24, -25, -26,           # right arm motor pos
                                -7,  -8,  -9,  -10, -11, -12, # left leg motor pos
                                -13, -14, -15, -16,           # left arm motor pos
                                -37, -38, -39, -40, -41, -42, # right leg motor vel
                                -43, -44, -45, -46,           # right arm motor vel
                                -27, -28, -29, -30, -31, -32, # left leg motor vel
                                -33, -34, -35, -36,           # left arm motor vel
                                -52, -53, -54, -55, -56,      # right joint pos
                                -47, -48, -49, -50, -51,      # left joint pos
                                -62, -63, -64, -65, -66,      # right joint vel
                                -57, -58, -59, -60, -61,      # left joint vel
                                ]

    # print(digit_state)
    mirror_vec = [-47, -48, -49, -50, -51]
    mirror_to_vec = [52, 53, 54, 55, 56]
    # Need to make every entry of this list to negative to get matched, even if they are in saggital plane.
    mirrored_s = mirror_tensor(torch.tensor(digit_state), mirror_vec)

    # Get actual positions
    s = np.take(digit_state, mirror_to_vec)
    print("mirroed s", mirrored_s)
    print("original s", s)
    print("mirroring difference norm", np.linalg.norm(mirrored_s - s))
    assert np.linalg.norm(mirrored_s - s) < 1e-2, \
        "Mirror incorrect"
