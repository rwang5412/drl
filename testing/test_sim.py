import numpy as np
import mujoco as mj

from sim import MjCassieSim, DigitMjSim
from .common import (
    DIGIT_MOTOR_NAME,
    DIGIT_JOINT_NAME,
    CASSIE_MOTOR_NAME,
    CASSIE_JOINT_NAME
)

OKGREEN = '\033[92m'
FAIL = '\033[91m'
ENDC = '\033[0m'

def test_all_sim():
    # TODO: Add other sims to this list after implemented
    sim_list = [MjCassieSim, DigitMjSim]
    num_pass = 0
    for sim in sim_list:
        print(f"Testing {sim.__name__}")
        num_pass += test_sim_init(sim)
        num_pass += test_sim_sim_forward(sim)
        num_pass += test_sim_viewer(sim)
        num_pass += test_sim_PD(sim)
        num_pass += test_sim_get_set(sim)
        num_pass += test_sim_indexes(sim)
        if num_pass == 6:
            print(f"{OKGREEN}{sim.__name__} passed all tests.{ENDC}")
        else:
            print(f"{FAIL}{sim.__name__} failed, only passed {num_pass} out of 6 tests.{ENDC}")
        num_pass = 0

def test_sim_init(sim):
    print("Making sim")
    test_sim = sim()
    test_sim.reset()
    print("Passed made sim")
    return True

def test_sim_sim_forward(sim):
    print("Testing sim forward")
    test_sim = sim()
    test_sim.reset()
    for i in range(100):
        test_sim.sim_forward()
    test_sim.sim_forward(dt = 0.5)
    print("Passed sim forward")
    return True

def test_sim_viewer(sim):
    print("Testing sim viewer, quit window to continue")
    test_sim = sim()
    test_sim.reset()
    test_sim.viewer_init()
    while test_sim.viewer.is_alive:
        if not test_sim.viewer.paused:
            for _ in range(50):
                test_sim.sim_forward()
        test_sim.viewer_render()
    print("Passed sim viewer")
    return True

def test_sim_PD(sim):
    print("Testing sim PD")
    test_sim = sim()
    test_sim.reset()
    test_sim.set_base_position(np.array([0, 0, 1.5]))
    test_sim.hold()
    for _ in range(1000):
        test_sim.sim_forward()
    test_sim.viewer_init()
    while test_sim.viewer.is_alive:
        for i in range(3000):
            test_sim.set_PD(test_sim.offset, np.zeros(test_sim.num_actuators), test_sim.kp, test_sim.kd)
            test_sim.sim_forward()
            if i%100==0 and test_sim.viewer.is_alive:
                test_sim.viewer_render()
    test_sim.release()
    if np.any((test_sim.get_motor_position() - test_sim.offset) > 1e-1):
        print(f"{FAIL}Failed sim PD test. Motor positions not close enough to target.{ENDC}")
        return False
    else:
        print("Passed sim PD")
        return True

def test_sim_get_set(sim):
    print("Testing sim getter and setter functions")
    test_sim = sim()
    test_sim.reset()
    # Test getters
    test_sim.get_joint_position()
    test_sim.get_joint_velocity()
    test_sim.get_base_position()
    test_sim.get_base_linear_velocity()
    test_sim.get_base_orientation()
    test_sim.get_base_angular_velocity()
    test_sim.get_torque()

    # Test setters
    test_sim.set_joint_position(np.zeros(test_sim.num_joints))
    test_sim.set_joint_velocity(np.zeros(test_sim.num_joints))
    test_sim.set_base_position(np.zeros(3))
    test_sim.set_base_linear_velocity(np.zeros(3))
    test_sim.set_base_orientation(np.zeros(4))
    test_sim.set_base_angular_velocity(np.zeros(3))
    test_sim.set_torque(np.zeros(test_sim.num_actuators))

    print("Pass sim getter and setter functions")
    return True

def test_sim_indexes(sim):
    test_sim = sim()
    test_sim.reset()

    motor_name = CASSIE_MOTOR_NAME if 'cassie' in sim.__name__.lower() else DIGIT_MOTOR_NAME
    joint_name = CASSIE_JOINT_NAME if 'cassie' in sim.__name__.lower() else DIGIT_JOINT_NAME

    motor_position_inds=[]
    motor_velocity_inds=[]
    for m in motor_name:
        motor_position_inds.append(test_sim.model.jnt_qposadr[mj.mj_name2id(test_sim.model, mj.mjtObj.mjOBJ_JOINT, m)])
        motor_velocity_inds.append(test_sim.model.jnt_dofadr[mj.mj_name2id(test_sim.model, mj.mjtObj.mjOBJ_JOINT, m)])

    joint_position_inds=[]
    joint_velocity_inds=[]
    for m in joint_name:
        joint_position_inds.append(test_sim.model.jnt_qposadr[mj.mj_name2id(test_sim.model, mj.mjtObj.mjOBJ_JOINT, m)])
        joint_velocity_inds.append(test_sim.model.jnt_dofadr[mj.mj_name2id(test_sim.model, mj.mjtObj.mjOBJ_JOINT, m)])

    assert motor_position_inds == test_sim.motor_position_inds, "Mismatch between motor_position_inds!"
    assert motor_velocity_inds == test_sim.motor_velocity_inds, "Mismatch between motor_velocity_inds!"
    assert joint_position_inds == test_sim.joint_position_inds, "Mismatch between joint_position_inds!"
    assert joint_velocity_inds == test_sim.joint_velocity_inds, "Mismatch between joint_velocity_inds!"

    print("Pass indices test")
    return True