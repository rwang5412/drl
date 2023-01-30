import mujoco as mj
import numpy as np
import time

from sim import (
    MjCassieSim,
    LibCassieSim,
    DigitMjSim,
    MujocoViewer,
)

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
    sim_list = [MjCassieSim, LibCassieSim, DigitMjSim]
    num_pass = 0
    for sim in sim_list:
        num_pass = 0
        print(f"Testing {sim.__name__}")
        # num_pass += test_sim_init(sim)
        # num_pass += test_sim_sim_forward(sim)
        # num_pass += test_sim_viewer(sim)
        num_pass += test_sim_glfw_multiple_viewer(sim)
        # num_pass += test_sim_PD(sim)
        # num_pass += test_sim_get_set(sim)
        # num_pass += test_sim_indexes(sim)
        if num_pass == 7:
            print(f"{OKGREEN}{sim.__name__} passed all tests.{ENDC}")
        else:
            print(f"{FAIL}{sim.__name__} failed, only passed {num_pass} out of 7 tests.{ENDC}")
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

def test_sim_glfw_multiple_viewer(sim):
    if "lib" in sim.__name__.lower():
        print("Bypass libcassie for dual window render.")
        return True
    print("Testing sim viewer, quit window to continue")
    test_sim = sim()
    test_sim.reset()
    test_sim.viewer_init(width=800, height=800)
    vis1 = test_sim.viewer
    # Create a second viewer that reads sim
    vis2 = MujocoViewer(test_sim.model, test_sim.data, test_sim.reset_qpos, \
        camera_id='forward-chest-realsense-d435/depth/image-rect', width=400, height=400)
    rs1 = vis1.render()
    rs2 = vis2.render()
    while rs1 and rs2:
        start_t = time.time()
        if not vis1.paused or not vis2.paused:
            for _ in range(50):
                test_sim.sim_forward()
        rs1 = vis1.render()
        rs2 = vis2.render()
        # Assume 2kHz sim for now
        delaytime = max(0, 50/2000 - (time.time() - start_t))
        time.sleep(delaytime)
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
    render_state = test_sim.viewer_render()
    while render_state:
        start_t = time.time()
        for _ in range(50):
            test_sim.set_PD(test_sim.offset, np.zeros(test_sim.num_actuators), test_sim.kp, test_sim.kd)
            test_sim.sim_forward()
        render_state = test_sim.viewer_render()
        delaytime = max(0, 50/2000 - (time.time() - start_t))
        time.sleep(delaytime)
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
    test_sim.set_base_position(np.ones(3))
    test_sim.set_base_linear_velocity(np.zeros(3))
    test_sim.set_base_orientation(np.array([0, 0.6987058, 0.2329019, 0.6764369]))
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
        motor_position_inds.append(test_sim.get_joint_qpos_adr(m))
        motor_velocity_inds.append(test_sim.get_joint_dof_adr(m))

    joint_position_inds=[]
    joint_velocity_inds=[]
    for m in joint_name:
        joint_position_inds.append(test_sim.get_joint_qpos_adr(m))
        joint_velocity_inds.append(test_sim.get_joint_dof_adr(m))

    assert motor_position_inds == test_sim.motor_position_inds, "Mismatch between motor_position_inds!"
    assert motor_velocity_inds == test_sim.motor_velocity_inds, "Mismatch between motor_velocity_inds!"
    assert joint_position_inds == test_sim.joint_position_inds, "Mismatch between joint_position_inds!"
    assert joint_velocity_inds == test_sim.joint_velocity_inds, "Mismatch between joint_velocity_inds!"

    print("Pass indices test")
    return True