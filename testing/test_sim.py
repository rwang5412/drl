import mujoco as mj
import numpy as np
import time

from sim import (
    MjCassieSim,
    LibCassieSim,
    MjDigitSim,
    MujocoViewer,
)

from .common import (
    DIGIT_MOTOR_NAME,
    DIGIT_JOINT_NAME,
    CASSIE_MOTOR_NAME,
    CASSIE_JOINT_NAME,
    CASSIE_NUM_BODY,
    CASSIE_NUM_GEOM,
    CASSIE_NV,
    DIGIT_NUM_BODY,
    DIGIT_NUM_GEOM,
    DIGIT_NV
)

from env.util.quaternion import quaternion2euler
from util.colors import FAIL, ENDC, OKGREEN

def test_all_sim():
    # TODO: Add other sims to this list after implemented
    # sim_list = [LibCassieSim, MjCassieSim, MjDigitSim]
    sim_list = [MjDigitSim]
    num_pass = 0
    failed = False
    for sim in sim_list:
        num_pass = 0
        print(f"Testing {sim.__name__}")
        test_sim_get_set(sim)
        # num_pass += test_sim_init(sim)
        # num_pass += test_sim_sim_forward(sim)
        # num_pass += test_sim_viewer(sim)
        # num_pass += test_sim_glfw_multiple_viewer(sim)
        # num_pass += test_sim_PD(sim)
        # num_pass += test_sim_get_set(sim)
        # num_pass += test_sim_indexes(sim)
        # num_pass += test_sim_body_pose(sim)
        # num_pass += test_sim_body_velocity(sim)
        # num_pass += test_sim_body_acceleration(sim)
        # num_pass += test_sim_body_contact_force(sim)
        # num_pass += test_sim_relative_pose(sim)
        if num_pass == 12:
            print(f"{OKGREEN}{sim.__name__} passed all tests.{ENDC}")
        else:
            failed = True
            print(f"{FAIL}{sim.__name__} failed, only passed {num_pass} out of 12 tests.{ENDC}")
        num_pass = 0
    if not failed:
        print(f"{OKGREEN}Passed all sim tests! \u2713{ENDC}")

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
    while rs1 or rs2:
        start_t = time.time()
        if rs1 and rs2:
            if not vis1.paused and not vis2.paused:
                for _ in range(50):
                    test_sim.sim_forward()
            rs1 = vis1.render()
            rs2 = vis2.render()
        elif rs1:
            if not vis1.paused:
                for _ in range(50):
                    test_sim.sim_forward()
            rs1 = vis1.render()
        elif rs2:
            if not vis1.paused:
                for _ in range(50):
                    test_sim.sim_forward()
            rs2 = vis2.render()
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

def test_sim_drop(sim):
    print("Testing sim PD")
    test_sim = sim()
    test_sim.reset()
    test_sim.set_base_position(np.array([0, 0, 1.5]))
    test_sim.viewer_init()
    render_state = test_sim.viewer_render()
    while render_state:
        start_t = time.time()
        if not test_sim.viewer_paused():
            for _ in range(50):
                test_sim.set_PD(test_sim.offset, np.zeros(test_sim.num_actuators), test_sim.kp, test_sim.kd)
                test_sim.sim_forward()
        render_state = test_sim.viewer_render()
        delaytime = max(0, 50/2000 - (time.time() - start_t))
        time.sleep(delaytime)
    return True

def test_sim_get_set(sim):
    print("Testing sim getter and setter functions")
    motor_name = CASSIE_MOTOR_NAME if 'cassie' in sim.__name__.lower() else DIGIT_MOTOR_NAME
    floor_name = "floor" if 'cassie' in sim.__name__.lower() else "ground"
    nbody = CASSIE_NUM_BODY if 'cassie' in sim.__name__.lower() else DIGIT_NUM_BODY
    nv = CASSIE_NV if 'cassie' in sim.__name__.lower() else DIGIT_NV
    ngeom = CASSIE_NUM_GEOM if 'cassie' in sim.__name__.lower() else DIGIT_NUM_GEOM
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
    test_sim.get_body_mass()
    test_sim.get_body_mass(name=motor_name[0])
    test_sim.get_dof_damping()
    test_sim.get_dof_damping(name=motor_name[0])
    test_sim.get_geom_friction()
    test_sim.get_geom_friction(name=floor_name)
    test_sim.get_body_ipos()
    test_sim.get_body_ipos(name=motor_name[0])

    # Test setters
    test_sim.set_joint_position(np.zeros(test_sim.num_joints))
    test_sim.set_joint_velocity(np.zeros(test_sim.num_joints))
    test_sim.set_base_position(np.ones(3))
    test_sim.set_base_linear_velocity(np.zeros(3))
    test_sim.set_base_orientation(np.array([0, 0.6987058, 0.2329019, 0.6764369]))
    test_sim.set_base_angular_velocity(np.zeros(3))
    test_sim.set_torque(np.zeros(test_sim.num_actuators))
    test_sim.set_body_mass(np.zeros(nbody))
    test_sim.set_body_mass(0, name=motor_name[0])
    test_sim.set_dof_damping(np.zeros(nv))
    test_sim.set_dof_damping(np.zeros(1), name=motor_name[0])
    test_sim.set_geom_friction(np.zeros((ngeom, 3)))
    test_sim.set_geom_friction(np.zeros(3), name=floor_name)
    test_sim.set_body_ipos(np.zeros((nbody, 3)))
    test_sim.set_body_ipos(np.zeros(3), name=motor_name[0])

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

def test_sim_body_pose(sim):
    test_sim = sim()
    test_sim.reset()
    x_target = np.array([0, 0, 1.5, 0, 0, 0, 1])
    test_sim.set_base_position(x_target[:3])
    test_sim.set_base_orientation(x_target[3:])
    test_sim.hold()
    test_sim.sim_forward(dt=0.1)
    x = test_sim.get_body_pose(name=test_sim.base_body_name)
    assert np.linalg.norm(x[:3] - x_target[:3]) < 1e-2, f"get_body_pose returns base at {x[:3]}, but sim sets to {x_target[:3]}."
    assert 1 - np.inner(x[3:], x_target[3:]) < 1e-2, f"get_body_pose returns base at {x[3:]}, but sim sets to {x_target[3:]}."

    test_sim.release()
    test_sim.sim_forward(dt=1)

    x_target = np.array([10, 10, 2, 0, 1, 0, 0])
    test_sim.set_base_position(x_target[:3])
    test_sim.set_base_orientation(x_target[3:])
    test_sim.hold()
    test_sim.sim_forward(dt=0.1)
    x = test_sim.get_body_pose(name=test_sim.base_body_name)
    assert np.linalg.norm(x[:3] - x_target[:3]) < 1e-2, f"get_body_pose returns base at {x[:3]}, but sim sets to {x_target[:3]}."
    assert 1 - np.inner(x[3:], x_target[3:]) < 1e-2, f"get_body_pose returns base at {x[3:]}, but sim sets to {x_target[3:]}."

    print("Passed sim get body pose")
    return True

def test_sim_body_velocity(sim):
    test_sim = sim()
    test_sim.reset()
    # NOTE: use small velocity here to avoid creating out of axis velocities, ie, if the torso has
    # inertia like Digit, set dtheta_x=1 will cause other axis to have velocities even sim 1 step.
    dx_target = np.array([1, 0, 0, 0, 0, 0])
    test_sim.set_base_linear_velocity(dx_target[:3])
    test_sim.set_base_angular_velocity(dx_target[3:])
    test_sim.sim_forward()
    dx = test_sim.get_body_velocity(name=test_sim.base_body_name)
    assert np.linalg.norm(dx[:3] - dx_target[:3]) < 1e-1, f"get_body_velocity returns base at {dx[:3]}, but sim sets to {dx_target[:3]}."
    assert np.linalg.norm(dx[3:] - dx_target[3:]) < 1e-1, f"get_body_velocity returns base at {dx[3:]}, but sim sets to {dx_target[3:]}."

    test_sim.release()
    test_sim.sim_forward(dt=1)

    dx_target = np.array([0, 0, 0.1, 0, 0, 0])
    test_sim.set_base_linear_velocity(dx_target[:3])
    test_sim.set_base_angular_velocity(dx_target[3:])
    test_sim.sim_forward()
    dx = test_sim.get_body_velocity(name=test_sim.base_body_name)
    assert np.linalg.norm(dx[:3] - dx_target[:3]) < 1e-1, f"get_body_velocity returns base at {dx[:3]}, but sim sets to {dx_target[:3]}."
    assert np.linalg.norm(dx[3:] - dx_target[3:]) < 1e-1, f"get_body_velocity returns base at {dx[3:]}, but sim sets to {dx_target[3:]}."

    print("Passed sim get body velocity")
    return True

def test_sim_body_acceleration(sim):
    test_sim = sim()
    test_sim.reset()
    test_sim.hold()
    test_sim.sim_forward(dt=1)
    ddx = test_sim.get_body_acceleration(name=test_sim.base_body_name)
    assert np.linalg.norm(ddx[:2]) < 1e-1, f"get_body_acceleration: robot should not have XY accelerations."
    assert np.abs(ddx[2] - 9.80665) < 1e-3, f"get_body_acceleration: gravity messed up."
    assert np.linalg.norm(ddx[3:]) < 1e-1, f"get_body_acceleration: robot should not have rotational accelerations."
    print("Passed sim get body acceleration")
    return True

def test_sim_body_contact_force(sim):
    """Hold robot in the air while feet touching ground. Check contact forces from each foot.
    Then drop the robot and check if floating base body gets contact forces.
    """
    test_sim = sim()
    test_sim.reset()
    # Slightly tilted down to let base falling to ground
    x_target = np.array([0, 0, 1, 0.9961947, 0, 0.0871557, 0])
    test_sim.set_base_position(x_target[:3])
    test_sim.set_base_orientation(x_target[3:])
    test_sim.hold()
    test_sim.sim_forward(dt=1)
    force = test_sim.get_body_contact_force(name=test_sim.feet_body_name[0])
    assert np.linalg.norm(force) > 10 , "get_body_contact_force returns wrong forces."
    force = test_sim.get_body_contact_force(name=test_sim.feet_body_name[1])
    assert np.linalg.norm(force) > 10 , "get_body_contact_force returns wrong forces."

    test_sim.release()
    test_sim.sim_forward(dt=2)
    force = test_sim.get_body_contact_force(name=test_sim.base_body_name)
    assert np.linalg.norm(force) > 10 , "get_body_contact_force returns wrong forces."

    print("Passed sim get body contact force")
    return True

def test_sim_relative_pose(sim):
    """Tilt torso/base + 10deg in pitch and measure feet flat (should be -10deg pitch)
    on ground angle diff in base frame.
    """
    test_sim = sim()
    test_sim.reset()
    # Slightly tilted down
    x_target = np.array([0, 0, 1, 0.9961947, 0, 0.0871557, 0])
    test_sim.set_base_position(x_target[:3])
    test_sim.set_base_orientation(x_target[3:])
    test_sim.hold()

    # Digit torso requires onger time to settle after torso/base joint set to 10deg
    dt = 5 if "digit" in sim.__name__.lower() else 1
    test_sim.sim_forward(dt=dt)
    p1 = test_sim.get_body_pose(name=test_sim.base_body_name)
    p2 = test_sim.get_site_pose(name=test_sim.feet_site_name[0])
    p3 = test_sim.get_site_pose(name=test_sim.feet_site_name[1])
    lfoot_in_base = test_sim.get_relative_pose(p1, p2)
    rfoot_in_base = test_sim.get_relative_pose(p1, p3)
    lfoot_euler = quaternion2euler(lfoot_in_base[3:7])/np.pi*180
    rfoot_euler = quaternion2euler(rfoot_in_base[3:7])/np.pi*180
    x_target_euler = quaternion2euler(x_target[3:7])/np.pi*180
    assert lfoot_euler[1] + x_target_euler[1] < 1e-1 , "get_relative_pose returns wrong lfoot angles."
    assert rfoot_euler[1] + x_target_euler[1] < 1e-1 , "get_relative_pose returns wrong rfoot angles."

    # NOTE: left for testing purposes
    # test_sim.viewer_init()
    # render_state = test_sim.viewer_render()
    # while render_state:
    #     start_t = time.time()
    #     for _ in range(50):
    #         x1 = test_sim.get_body_pose(name=test_sim.base_body_name)
    #         x2 = test_sim.get_site_pose(name=test_sim.feet_site_name[0])
    #         x3 = test_sim.get_relative_pose(x1, x2)
    #         print("left foot relative to base angles, ", quaternion2euler(x3[3:7])/np.pi*180)
    #         print("left foot relative to base positions, ", x3[:3])
    #         print()
    #         test_sim.sim_forward()
    #     render_state = test_sim.viewer_render()
    #     delaytime = max(0, 50/2000 - (time.time() - start_t))
    #     time.sleep(delaytime)

    print("Passed sim get_relative_pose")
    return True

def test_sim_dr(sim):
    test_sim = sim()
    # print(test_sim.get_dof_damping(name="left-hip-roll"))
    # print(test_sim.get_dof_damping(name="left-achilles-rod"))
    # foo = test_sim.get_body_mass(name="left-hip-roll")
    foo = test_sim.get_dof_damping(name="left-hip-roll")
    print(foo)
    foo = 5
    print(test_sim.get_dof_damping(name="left-hip-roll"))
    test_sim.set_geom_friction(np.zeros(10), name="floor")
    exit()
    # print(test_sim.get_body_mass(name="left-hip-roll"))
    foo = test_sim.get_joint_position()
    # print(foo)
    foo[0] = 5
    # print(test_sim.get_joint_position())