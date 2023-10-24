import agility
import agility.messages as msg
import atexit
import argparse
import asyncio
import datetime
import numpy as np
import sys
import os
import platform
import pickle
import select
import signal
import sys
import termios
import time
import torch
import tty

from env.util.quaternion import mj2scipy
from multiprocessing import Process
from scipy.spatial.transform import Rotation as R
from sim.digit_sim.digit_ar_sim.digit_udp import DigitUdp
from sim.digit_sim.digit_ar_sim.interface_ctypes import *
from testing.common import (
    DIGIT_MOTOR_MJ2LLAPI_INDEX,
    MOTOR_POSITION_SET
)
from util.colors import WARNING, ENDC
from util.digit_topic import DigitStateTopic
from util.env_factory import add_env_parser, env_factory
from util.nn_factory import load_checkpoint, nn_factory

def save_log():
    global logdir, log_size, part_num, log_ind, time_log, input_log, output_log, orient_add_log

    filename = os.path.join(logdir, f"logdata_part{part_num}.pkl")
    print("Logging to {}".format(filename))
    # Truncate log data to actual size
    if log_ind < log_size:
        for key, val in input_log.items():
            input_log[key] = val[:log_ind]
        for key, val in output_log.items():
            output_log[key] = val[:log_ind]

    data = {"time": time_log[:log_ind],
            "output": output_log,
            "input": input_log,
            "orient_add": orient_add_log[:log_ind]}
    with open(filename, "wb") as filep:
        pickle.dump(data, filep)
    part_num += 1

def close_ar_sim(ar_sim):
    ar_sim.close()

async def run(actor, env, do_log = True, pol_name = "test"):

    global log_size, log_ind, part_num, time_log, input_log, output_log, orient_add_log

    # Start ar-control
    ar_control_path = os.path.expanduser("~/ar-software-2023.01.13a/ar-software/ar-control")
    toml_path = os.path.abspath("./sim/digit_sim/digit_ar_sim/llapi/digit-rl.toml")
    if os.path.isfile(ar_control_path):
        print("Starting ar-control")
        ar_sim = agility.Simulator(ar_control_path, toml_path)
    else:
        print(f"{WARNING}Assuming ar-control already running{ENDC}")

    save_log_p = None   # Save log process for async file saving
    if do_log:
        atexit.register(save_log)
    atexit.register(close_ar_sim, ar_sim)

    if platform.node() == "digit-nuc":
        ROBOT_ADDRESS = '10.10.1.1'
    else:
        ROBOT_ADDRESS = '127.0.0.1'

    api = agility.JsonApi(address=ROBOT_ADDRESS,
                          port=8080,
                          connect_timeout=1)
    try:
        await api.connect()
    except:
        raise Exception(f"Cannot connect api at {ROBOT_ADDRESS} with port {8080}\n"
                        f"Check if simulator is running and port is correct!")

    digit_udp = DigitUdp(robot_address=ROBOT_ADDRESS)
    topic = DigitStateTopic(digit_udp)

    env.reset_for_test()
    cmd_policy = llapi_command_pd_t()
    for i in range(NUM_MOTORS):
        cmd_policy.kp[i] = env.kp[DIGIT_MOTOR_MJ2LLAPI_INDEX[i]]
        cmd_policy.kd[i] = env.kd[DIGIT_MOTOR_MJ2LLAPI_INDEX[i]]
        cmd_policy.feedforward_torque[i] = 0.0

    query_json = True
    api_mode = "locomotion"

    cmd_stand = llapi_command_pd_t()
    for i in range(NUM_MOTORS):
        cmd_stand.kp[i] = 1000
        cmd_stand.kd[i] = 30
        cmd_stand.position[i] = MOTOR_POSITION_SET['stand'][i]

    await api.send(["start-log-session", {"label": pol_name}])

    init_llapi_time = time.perf_counter()
    obs = None
    while obs is None and time.perf_counter() - init_llapi_time < 1.5:
        digit_udp.send_pd(llapi_command_pd_t())
        obs = topic.recv()
        if time.perf_counter() - init_llapi_time > 1:
            raise Exception("Cannot receive observation from simulator!")
    print("Connected Python with LLAPI and Digit!")
    for _ in range(50):
        obs = topic.recv()
        q = np.array([obs.base.orientation.w,
                      obs.base.orientation.x,
                      obs.base.orientation.y,
                      obs.base.orientation.z])
        euler = R.from_quat(mj2scipy(q)).as_euler('xyz')
        env.orient_add = euler[2]

    def isData():
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
    old_settings = termios.tcgetattr(sys.stdin)

    env.display_controls_menu()
    env.display_control_commands()
    print(f"\033[{env.num_menu_backspace_lines}B\033[K", end='\r')
    loop_start_time = time.perf_counter()
    pol_time = time.perf_counter()
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            obs = topic.recv()
            # Dummy json call to keep json alive. Only do once per policy call
            if query_json:
                await api.query(msg.GetRobotInfo())
                query_json = False

            if isData():
                c = sys.stdin.read(1)
                print(f"\033[{env.num_menu_backspace_lines}A\033[K", end='\r')
                env.interactive_control(c)
                print(f"\033[{env.num_menu_backspace_lines}B\033[K", end='\r')
                if c == "m":  # Toggle api mode
                    if api_mode == "locomotion":
                        api_mode = "llapi"
                        await api.request_privilege('change-action-command')
                        await api.send(msg.ActionSetOperationMode(mode="low-level-api"))
                        q = np.array([obs.base.orientation.w,
                                      obs.base.orientation.x,
                                      obs.base.orientation.y,
                                      obs.base.orientation.z])
                        euler = R.from_quat(mj2scipy(q)).as_euler('xyz')
                        env.orient_add = euler[2]
                    elif api_mode == "llapi":
                        api_mode = "locomotion"
                        await api.request_privilege('change-action-command')
                        await api.send(msg.ActionSetOperationMode(mode="locomotion"))

            update_time = time.perf_counter() - pol_time
            if update_time >= 1 / env.default_policy_rate:

                query_json = True
                env.llapi_obs = obs
                robot_state = env.get_robot_state()
                RL_state = env.get_state()

                with torch.no_grad():
                    action = actor(torch.tensor(RL_state).float(), deterministic=True).numpy()
                action_sum = env.offset + action
                for i in range(NUM_MOTORS):
                    cmd_policy.position[i] = action_sum[DIGIT_MOTOR_MJ2LLAPI_INDEX[i]]
                digit_udp.send_pd(cmd_policy)
                env.hw_step()
                print(f"mode: {api_mode}, "
                      f"delay: {((time.perf_counter() - pol_time) - (1 / env.default_policy_rate)) * 100:.3f} ms", end="\r")
                pol_time = time.perf_counter()

                if do_log:
                    time_log[log_ind] = pol_time
                    for i in range(len(env.robot_state_names)):
                        input_log[env.robot_state_names[i]][log_ind] = robot_state[i]
                    for i in range(len(env.extra_input_names)):
                        input_log[env.extra_input_names[i]][log_ind] = RL_state[len(env.robot_state_names) + i]
                    for i in range(len(env.output_names)):
                        output_log[env.output_names[i]][log_ind] = action_sum[i]
                    orient_add_log[log_ind] = env.orient_add
                    log_ind += 1
                    if log_ind >= log_size:
                        if save_log_p is not None:
                            save_log_p.join()
                        save_log_p = Process(target=save_log)
                        log_ind = 0
                        part_num += 1

            delaytime = 1/1000 - (time.perf_counter() - loop_start_time)
            while delaytime > 0:
                t0 = time.perf_counter()
                time.sleep(1e-5)
                delaytime -= time.perf_counter() - t0
                loop_start_time = time.perf_counter()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, help="path to folder containing policy and run details")
    parser.add_argument("--no-log", dest='do_log', default=True, action="store_false", help="Whether to log data or not. True by default")

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
    add_env_parser(previous_args_dict['all_args'].env_name, parser, is_eval=True)
    args = parser.parse_args()

    # Overwrite previous env args with current input
    for arg, val in vars(args).items():
        if hasattr(previous_args_dict['env_args'], arg):
            setattr(previous_args_dict['env_args'], arg, val)

    # Load environment
    previous_args_dict['env_args'].simulator_type = "ar_async"
    if hasattr(previous_args_dict['env_args'], 'velocity_noise'):
        delattr(previous_args_dict['env_args'], 'velocity_noise')
    if hasattr(previous_args_dict['env_args'], 'state_est'):
        delattr(previous_args_dict['env_args'], 'state_est')
    env = env_factory(previous_args_dict['all_args'].env_name, previous_args_dict['env_args'])()

    # Load model class and checkpoint
    actor, critic = nn_factory(args=previous_args_dict['nn_args'], env=env)
    load_checkpoint(model=actor, model_dict=actor_checkpoint)
    actor.eval()
    actor.training = False
    if hasattr(actor, 'init_hidden_state'):
        actor.init_hidden_state()

    # Global data for logging
    global log_size
    global log_ind
    global part_num
    global time_log # time stamp
    global input_log # network inputs
    global output_log # network outputs
    global orient_add_log # heading offset
    global final_save
    log_size = 100000
    log_ind = 0
    part_num = 0
    time_log   = [time.time()] * log_size # time stamp
    input_log = {} # network inputs
    final_save = False
    for name in env.robot_state_names + env.extra_input_names:
        input_log[name] = [0.0] * log_size
    output_log = {}
    for motor in env.output_names:
        output_log[motor] = [0.0] * log_size
    orient_add_log = [0.0] * log_size # heading offset

    global logdir
    LOG_NAME = args.path.rsplit('/', 3)[1]
    directory = os.path.dirname(os.path.realpath(__file__)) + "/hardware_logs/digit/"
    timestr = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    logdir = os.path.join(directory, LOG_NAME, timestr)
    # Check if output directory already exists. If it does, increment logdir name
    index = ''
    while os.path.exists(logdir + index):
        if index:
            index = '_(' + str(int(index[2:-1]) + 1) + ')'
        else:
            index = '_(1)'
    logdir += index + "/"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    asyncio.run(run(actor, env, do_log=args.do_log, pol_name = LOG_NAME))