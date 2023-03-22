import argparse, time, pickle, platform
import os, sys, datetime
import select, termios, tty, atexit
from math import floor

import numpy as np
import torch
from multiprocessing import Process
from sim.cassie_sim.cassiemujoco.cassieUDP import *
from sim.cassie_sim.cassiemujoco.cassiemujoco_ctypes import *
from env.util.quaternion import (
    euler2quat,
    inverse_quaternion,
    rotate_by_quaternion,
    quaternion_product,
    quaternion2euler
)

from util.nn_factory import load_checkpoint, nn_factory
from util.env_factory import env_factory

# entry file for run a specified udp setup
# cassie-async (sim), digit-ar-control-async (sim), cassie-real, digit-real

def save_log():
    global log_hf_ind, log_lf_ind, logdir, part_num, sto_num, time_hf_log, output_log, state_log, target_log, speed_log, orient_log, phaseadd_log, time_lf_log, input_log

    filename = "logdata_part" + str(part_num) + "_sto" + str(sto_num) + ".pkl"
    filename = os.path.join(logdir, filename)
    print("Logging to {}".format(filename))
    print("exit at time {}".format(time_hf_log[log_hf_ind-1]))
    print("save log: log_hf_ind {}".format(log_hf_ind))
    data = {"highfreq": True,
            "time_hf": time_hf_log[:log_hf_ind],
            "time_lf": time_lf_log[:log_lf_ind],
            "output": output_log[:log_hf_ind],
            "input": input_log[:log_lf_ind],
            "state": state_log[:log_hf_ind],
            "target": target_log[:log_hf_ind],
            "speed": speed_log[:log_hf_ind],
            "orient": orient_log[:log_hf_ind],
            "phase_add": phaseadd_log[:log_hf_ind],
            "simrate": 50}
    with open(filename, "wb") as filep:
        pickle.dump(data, filep)
    part_num += 1

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

# 2 kHz execution : PD control with or without baseline action
def PD_step(cassie_udp, cassie_env, action):
    target = action[:] + cassie_env.offset

    u = pd_in_t()
    for i in range(5):
        u.leftLeg.motorPd.pGain[i]  = cassie_env.kp[i]
        u.rightLeg.motorPd.pGain[i] = cassie_env.kp[i + 5]

        u.leftLeg.motorPd.dGain[i]  = cassie_env.kd[i]
        u.rightLeg.motorPd.dGain[i] = cassie_env.kd[i + 5]

        u.leftLeg.motorPd.torque[i]  = 0  # Feedforward torque
        u.rightLeg.motorPd.torque[i] = 0

        u.leftLeg.motorPd.pTarget[i]  = target[i]
        u.rightLeg.motorPd.pTarget[i] = target[i + 5]

        u.leftLeg.motorPd.dTarget[i]  = 0
        u.rightLeg.motorPd.dTarget[i] = 0

    cassie_udp.send_pd(u)

    # return log data
    return target

def execute(policy, env, args, do_log, exec_rate=1):
    global log_size, log_hf_ind, log_lf_ind, part_num, sto_num, save_dict, time_hf_log, output_log, state_log, target_log, speed_log, orient_log, phaseadd_log, time_lf_log, input_log

    # Determine whether running in simulation or on the robot
    if platform.node() == 'cassie':
        cassieudp = CassieUdp(remote_addr='10.10.10.3', remote_port='25010',
                        local_addr='10.10.10.100', local_port='25011')
    else:
        cassieudp = CassieUdp()  # local testing

    if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

    if exec_rate > env.default_policy_rate:
        print("Error: Execution rate can not be greater than simrate")
        exit()
    # Lock exec_rate to even dividend of simrate
    rem = env.default_policy_rate // exec_rate
    exec_rate = env.default_policy_rate // rem
    print("Execution rate: {} ({:.2f} Hz)".format(exec_rate, 2000/exec_rate))

    # ESTOP position. True means ESTOP enabled and robot is not running.
    STO = False
    logged = False
    part_num = 0
    sto_num = 0
    save_log_p = None
    env.reset()
    env.turn_rate = 0
    env.y_velocity = 0
    env.x_velocity = 0
    env.clock._phase = 0
    env.clock._cycle_time = 0.8
    env.clock._swing_ratios = [0.4, 0.4]
    env.clock._period_shifts = [0, 0.5]

    # 0: walking
    # 1: standing
    # 2: damping
    action = None
    operation_mode = 0
    D_mult = 1  # Reaaaaaally bad stability problems if this is pushed higher as a multiplier
                # Might be worth tuning by joint but something else if probably needed

    empty_u = pd_in_t()
    damp_u = pd_in_t()
    for i in range(5):
        empty_u.leftLeg.motorPd.pGain[i] = 0.0
        empty_u.leftLeg.motorPd.dGain[i] = 0.0
        empty_u.rightLeg.motorPd.pGain[i] = 0.0
        empty_u.rightLeg.motorPd.dGain[i] = 0.0
        empty_u.leftLeg.motorPd.pTarget[i] = 0.0
        empty_u.rightLeg.motorPd.pTarget[i] = 0.0

        damp_u.leftLeg.motorPd.pGain[i] = 0.0
        damp_u.leftLeg.motorPd.dGain[i] = D_mult*env.kd[i]
        damp_u.rightLeg.motorPd.pGain[i] = 0.0
        damp_u.rightLeg.motorPd.dGain[i] = D_mult*env.kd[i + 5]
        damp_u.leftLeg.motorPd.pTarget[i] = 0.0
        damp_u.rightLeg.motorPd.pTarget[i] = 0.0

    old_settings = termios.tcgetattr(sys.stdin)
    count = 0
    pol_time = 0
    state_count = 0

    # Connect to the simulator or robot
    print('Connecting...')
    state = None
    while state is None:
        cassieudp.send_pd(pd_in_t())
        time.sleep(0.001)
        state = cassieudp.recv_newest_pd()
    received_data = True
    print('Connected!\n')

    try:
        tty.setcbreak(sys.stdin.fileno())

        t = time.monotonic()
        t0 = t
        lt = 0
        pol_time = 0
        first = True
        while True:

            # Get newest state
            t = time.monotonic()
            state = cassieudp.recv_newest_pd()
            while state is None:
                state_count += 1
                time.sleep(0.0001*exec_rate)
                state = cassieudp.recv_newest_pd()

            # No continue
            if platform.node() == 'cassie':
                # Control with Taranis radio controller
                if state.radio.channel[9] < -0.5:
                    operation_mode = 2  # down -> damping
                elif state.radio.channel[9] > 0.5:
                    operation_mode = 1  # up -> nothing
                else:
                    operation_mode = 0  # mid -> normal walking

                # Reset orientation on STO
                if state.radio.channel[8] < 0:
                    STO = True
                    env.robot_state = state
                    env.orient_add = quaternion2euler(env.sim.robot_state.pelvis.orientation[:])[2]
                else:
                    STO = False
                    logged = False

                # Orientation control (Do manually instead of turn_rate)
                env.turn_rate = state.radio.channel[3] * np.pi/8
                # X and Y speed control
                env.x_velocity += state.radio.channel[0] / (60.0*env.default_policy_rate)
                env.y_velocity += state.radio.channel[1] / (60.0*env.default_policy_rate)
                # Example of setting things manually instead. Reference to what radio channel corresponds to what joystick/knob:
                # https://github.com/agilityrobotics/cassie-doc/wiki/Radio#user-content-input-configuration
                # env.cmd_dict['step_freq'] = 1 + state.radio.channel[5]
                # ratio = 0.5 + state.radio.channel[6] / 2
                # env.cmd_dict['ratio'] = [ratio, 1-ratio]
                # env.cmd_dict['period_shift'] = [0, (state.radio.channel[7]+1)/2]

            else:
                """
                    Control of the robot in simulation using a keyboard
                """

                if isData():
                    c = sys.stdin.read(1)
                    if c == 'x':
                        if hasattr(policy, 'init_hidden_state'):
                            policy.init_hidden_state()
                    elif c == 't':
                        STO = True
                        print("\nESTOP enabled")
                    else:
                        env.interactive_control(c)

            env.x_velocity = np.clip(env.x_velocity, args.min_x, args.max_x)
            env.y_velocity = np.clip(env.y_velocity, args.min_y, args.max_y)

            if STO:
                if not logged:
                    logged = True
                    save_log()
                    sto_num += 1
                    part_num = 0
                    log_hf_ind = 0
                    log_lf_ind = 0

            curr_state = state
            # Continue to update state while sleeping to hit desired script frequency
            while state is None or time.monotonic() - t < exec_rate/2000:
                state_count += 1
                time.sleep(0.0001*exec_rate)
                curr_state = cassieudp.recv_newest_pd()
                if curr_state:
                    state = curr_state

            #------------------------------- Normal Walking ---------------------------
            if operation_mode == 0:
                count += 1
                update_time = time.monotonic() - pol_time

                if first or update_time > 1 / env.default_policy_rate:

                    lt = 0
                    new_time = time.time()
                    """
                        Low frequency (40 Hz) Section. Update policy action
                    """
                    env.sim.robot_state = state
                    RL_state = env.get_state()
                    with torch.no_grad():
                        action = policy(torch.tensor(RL_state).float(), deterministic=True).numpy()
                    target = PD_step(cassieudp, env, action)
                    pol_time = time.monotonic()
                    # Update env quantities
                    env.orient_add += env.turn_rate / (2000 / 50)
                    env.clock.increment()

                    if do_log:
                        time_lf_log[log_lf_ind] = time.time()
                        input_log[log_lf_ind] = RL_state
                        target_log[log_lf_ind] = target
                        log_lf_ind += 1

                    # Measure delay
                    # print('delay: {:6.1f} ms'.format((time.monotonic() - t) * 1000))
                    # print("compute time:", (time.time() - new_time)*1000)
                    measured_delay = (update_time - 1 / env.default_policy_rate) * 1000
                    if not first:
                        sys.stdout.write("Speed: {:.2f}\t count : {:02d}/{:02d} \tdelay: {:2.2f} ms\r".format(env.x_velocity, count, env.default_policy_rate//exec_rate, measured_delay))
                        sys.stdout.flush()
                    first = False
                    count = 0
                    # pol_time = new_time


                """
                    High frequency (2000 Hz) Section
                """

                if do_log:
                    time_hf_log[log_hf_ind] = time.time()
                    output_log[log_hf_ind] = action
                    state_log[log_hf_ind] = state
                    speed_log[log_hf_ind] = env.x_velocity
                    orient_log[log_hf_ind] = env.orient_add
                    phaseadd_log[log_hf_ind] = env.clock._cycle_time
                    log_hf_ind += 1

                if log_hf_ind == log_size and do_log:
                    if save_log_p is not None:
                        save_log_p.join()
                    save_log_p = Process(target=save_log)
                    save_log_p.start()
                    part_num += 1
                    log_hf_ind = 0
                    log_lf_ind = 0

            #------------------------------- Empty Action ---------------------------
            elif operation_mode == 1:
                print('Applying no action')
                # Do nothing
                cassieudp.send_pd(empty_u)

            #------------------------------- Shutdown Damping ---------------------------
            elif operation_mode == 2:
                print('Shutdown Damping. Multiplier = ' + str(D_mult))
                cassieudp.send_pd(damp_u)

            #---------------------------- Other, should not happen -----------------------
            else:
                print('Error, In bad operation_mode with value: ' + str(operation_mode))


    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=None, help="path to folder containing policy and run details")
parser.add_argument("--exec_rate", default=1, type=int, help="Controls the execution rate of the script. Is 1 (full 2kHz) be default")
parser.add_argument("--no_log", dest='do_log', default=True, action="store_false", help="Whether to log data or not. True by default")
parser.add_argument("--max_x", default=4.0, type=float, help="Maximum x speed")
parser.add_argument("--min_x", default=0.0, type=float, help="Minimum x speed")
parser.add_argument("--max_y", default=0.5, type=float, help="Maximum y speed")
parser.add_argument("--min_y", default=-0.5, type=float, help="Minimum y speed")

# Manually handle path argument
try:
    path_idx = sys.argv.index("--path")
    model_path = sys.argv[path_idx + 1]
    if not isinstance(model_path, str):
        print(f"{__file__}: error: argument --path received non-string input.")
        sys.exit()
except ValueError:
    print(f"No path input given. Usage is 'python eval.py simple --path /path/to/policy'")

previous_args_dict = pickle.load(open(os.path.join(model_path, "experiment.pkl"), "rb"))
actor_checkpoint = torch.load(os.path.join(model_path, 'actor.pt'), map_location='cpu')
args = parser.parse_args()

# Load environment
previous_args_dict['env_args'].simulator_type = "libcassie"
previous_args_dict['env_args'].state_est = True
env = env_factory(previous_args_dict['all_args'].env_name, previous_args_dict['env_args'])()

# Load model class and checkpoint
actor, critic = nn_factory(args=previous_args_dict['nn_args'], env=env)
load_checkpoint(model=actor, model_dict=actor_checkpoint)
actor.eval()
actor.training = False

LOG_NAME = os.path.basename(os.path.normpath(args.path))
directory = os.path.dirname(os.path.realpath(__file__)) + "/hardware_logs/"
filename = "logdata"
timestr = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M') + "/"
if not os.path.exists(directory + timestr + LOG_NAME + "/"):
    os.makedirs(directory + timestr + LOG_NAME + "/")
logdir = directory + timestr + LOG_NAME + "/"
filename = directory + timestr + LOG_NAME + "/" + filename + ".pkl"

# Global data for logging
log_size = 100000
log_lf_ind = 0
log_hf_ind = 0
time_lf_log   = [time.time()] * log_size # time stamp
time_hf_log   = [time.time()] * log_size # time stamp
input_log  = [np.ones(actor.obs_dim)] * log_size # network inputs
output_log = [np.ones(actor.action_dim)] * log_size # network outputs
state_log  = [state_out_t()] * log_size  # cassie state
target_log = [np.ones(10)] * log_size  # PD target log
speed_log  = [0.0] * log_size # speed input commands
orient_log  = [0.0] * log_size # orient input commands
phaseadd_log  = [0.0] * log_size # frequency input commands

part_num = 0
sto_num = 0

if args.do_log:
    atexit.register(save_log)
execute(actor, env, args, args.do_log, args.exec_rate)