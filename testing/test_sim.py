import numpy as np
import time

from sim.cassie_sim import MjCassieSim
from sim.cassie_sim import LibCassieSim

OKGREEN = '\033[92m'
FAIL = '\033[91m'
ENDC = '\033[0m'

def test_mj_sim():
    mj_sim = MjDigitSim()
    mj_sim.viewer_init()
    while mj_sim.viewer.is_alive:
        if not mj_sim.viewer.paused:
            for _ in range(50):
                mj_sim.sim_forward()
        mj_sim.viewer_render()

def test_all_sim():
    # TODO: Add other sims to this list after implemented
    sim_list = [MjCassieSim, LibCassieSim]
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'
    for sim in sim_list:
        num_pass = 0
        print(f"Testing {sim.__name__}")
        num_pass += test_sim_init(sim)
        num_pass += test_sim_sim_forward(sim)
        num_pass += test_sim_viewer(sim)
        num_pass += test_sim_PD(sim)
        num_pass += test_sim_get_set(sim)
        if num_pass == 5:
            print(f"{OKGREEN}{sim.__name__} passed all tests.{ENDC}")
        else:
            print(f"{FAIL}{sim.__name__} failed, only passed {num_pass} out of 5 tests.{ENDC}")

def test_sim_init(sim):
    print("Making sim")
    test_sim = sim()
    print("Passed made sim")
    return True

def test_sim_sim_forward(sim):
    print("Testing sim forward")
    test_sim = sim()
    for i in range(100):
        test_sim.sim_forward()
    test_sim.sim_forward(dt = 0.5)
    print("Passed sim forward")
    return True

def test_sim_viewer(sim):
    print("Testing sim viewer, quit window to continue")
    test_sim = sim()
    test_sim.viewer_init()
    render_state = test_sim.viewer_render()
    while render_state:
        start_t = time.time()
        if not test_sim.viewer_paused():
            for _ in range(50):
                test_sim.sim_forward()
        render_state = test_sim.viewer_render()
        # Assume 2kHz sim for now
        delaytime = max(0, 50/2000 - (time.time() - start_t))
        time.sleep(delaytime)
    print("Passed sim viewer")
    return True

def test_sim_PD(sim):
    print("Testing sim PD")
    test_sim = sim()
    test_sim.set_com_pos(np.array([0, 0, 1.5]))
    test_sim.hold()
    for _ in range(1000):
        test_sim.sim_forward()
    for _ in range(3000):
        test_sim.set_PD(test_sim.offset, np.zeros(10), test_sim.kp, test_sim.kd)
        test_sim.sim_forward()
    test_sim.release()
    if np.any((test_sim.get_motor_pos() - test_sim.offset) > 1e-1):
        print(f"{FAIL}Failed sim PD test. Motor positions not close enough to target.{ENDC}")
        return False
    else:
        print("Passed sim PD")
        return True

def test_sim_get_set(sim):
    print("Testing sim getter and setter functions")
    test_sim = sim()
    # Test getters
    test_sim.get_joint_pos()
    test_sim.get_joint_vel()
    test_sim.get_motor_pos()
    test_sim.get_motor_vel()
    test_sim.get_com_pos()
    test_sim.get_com_trans_vel()
    test_sim.get_com_quat()
    test_sim.get_com_rot_vel()
    test_sim.get_torque()

    # Test setters
    test_sim.set_joint_pos(np.zeros(test_sim.num_joint))
    test_sim.set_joint_vel(np.zeros(test_sim.num_joint))
    test_sim.set_motor_pos(np.zeros(test_sim.num_actuators))
    test_sim.set_motor_vel(np.zeros(test_sim.num_actuators))
    test_sim.set_com_pos(2*np.ones(3))
    test_sim.set_com_trans_vel(np.zeros(3))
    test_sim.set_com_quat(np.array([0, 0.6987058, 0.2329019, 0.6764369]))
    test_sim.set_com_rot_vel(np.zeros(3))
    test_sim.set_torque(np.zeros(test_sim.num_actuators))

    print("Pass sim getter and setter functions")
    return True
