import numpy as np

from sim.cassie_sim import MjCassieSim
from sim.digit_sim import DigitMjSim

OKGREEN = '\033[92m'
FAIL = '\033[91m'
ENDC = '\033[0m'

# self.motor_name = ['left-leg/hip-roll', 'left-leg/hip-yaw', 'left-leg/hip-pitch',
#                        'left-leg/knee', 'left-leg/toe-a', 'left-leg/toe-b',
#                        'left-arm/shoulder-roll','left-arm/shoulder-pitch', 'left-arm/shoulder-yaw', 'left-arm/elbow',
#                        'right-leg/hip-roll', 'right-leg/hip-yaw', 'right-leg/hip-pitch',
#                        'right-leg/knee', 'right-leg/toe-a', 'right-leg/toe-b',
#                        'right-arm/shoulder-roll','right-arm/shoulder-pitch', 'right-arm/shoulder-yaw', 'right-arm/elbow']

# self.joint_name = ['left-leg/shin', 'left-leg/tarsus', 'left-leg/heel-spring', 'left-leg/toe-pitch', 'left-leg/toe-roll',
#                     'right-leg/shin', 'right-leg/tarsus', 'right-leg/heel-spring', 'right-leg/toe-pitch', 'right-leg/toe-roll']
    
# self.motor_position_inds=[]
# self.motor_velocity_inds=[]
# for m in self.motor_name:
#     self.motor_position_inds.append(self.model.jnt_qposadr[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, m)])
#     self.motor_velocity_inds.append(self.model.jnt_dofadr[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, m)])
#     print(m, mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, m), self.model.jnt_qposadr[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, m)])
# print(self.motor_position_inds)
# print(self.motor_velocity_inds)
# self.joint_position_inds=[]
# self.joint_velocity_inds=[]
# for m in self.joint_name:
#     self.joint_position_inds.append(self.model.jnt_qposadr[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, m)])
#     self.joint_velocity_inds.append(self.model.jnt_dofadr[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, m)])
#     print(m, mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, m), self.model.jnt_qposadr[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, m)])
# print(self.joint_position_inds)
# print(self.joint_velocity_inds)
# exit()

def test_mj_sim():
    mj_sim = DigitMjSim()
    mj_sim.hold()
    mj_sim.viewer_init()
    mj_sim.viewer.paused = True
    while mj_sim.viewer.is_alive:
        if not mj_sim.viewer.paused:
            for _ in range(50):
                mj_sim.sim_forward()
        mj_sim.viewer_render()

def test_all_sim():
    # TODO: Add other sims to this list after implemented
    # sim_list = [MjCassieSim]
    # num_pass = 0
    # for sim in sim_list:
    #     print(f"Testing {sim.__name__}")
    #     num_pass += test_sim_init(sim)
    #     num_pass += test_sim_sim_forward(sim)
    #     num_pass += test_sim_viewer(sim)
    #     num_pass += test_sim_PD(sim)
    #     num_pass += test_sim_get_set(sim)
    #     if num_pass == 5:
    #         print(f"{OKGREEN}{sim.__name__} passed all tests.{ENDC}")
    #     else:
    #         print(f"{FAIL}{sim.__name__} failed, only passed {num_pass} out of 5 tests.{ENDC}")

    test_mj_sim()
    
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
    test_sim.set_com_pos(np.array([0, 0, 1.5]))
    test_sim.hold()
    for _ in range(1000):
        test_sim.sim_forward()
    for _ in range(3000):
        test_sim.set_PD(test_sim.offset, np.zeros(10), test_sim.P, test_sim.D)
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
    test_sim.get_com_pos()
    test_sim.get_com_trans_vel()
    test_sim.get_com_quat()
    test_sim.get_com_rot_vel()
    test_sim.get_torque()

    # Test setters
    test_sim.set_joint_pos(np.zeros(test_sim.num_joint))
    test_sim.set_joint_vel(np.zeros(test_sim.num_joint))
    test_sim.set_com_pos(np.zeros(3))
    test_sim.set_com_trans_vel(np.zeros(3))
    test_sim.set_com_quat(np.zeros(4))
    test_sim.set_com_rot_vel(np.zeros(3))
    test_sim.set_torque(np.zeros(test_sim.num_actuators))

    print("Pass sim getter and setter functions")
    return True
