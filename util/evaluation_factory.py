import torch
import time
import argparse
import sys, os 
import numpy as np

from util.env_factory import env_factory
from util.drivers import Keyboard

def simple_eval(actor, env, episode_length_max=300):
    """Simply evaluating policy in visualization window and no user input 

    Args:
        actor: Actor loaded outside this function. If Actor is None, this function will evaluate
            noisy actions without any policy.
        env: Environment instance for actor
        episode_length_max (int, optional): Max length of episode for evaluation. Defaults to 500.
    """
    with torch.no_grad():
        state = env.reset()
        done = False
        episode_length = 0
        episode_reward = []

        if hasattr(actor, 'init_hidden_state'):
            actor.init_hidden_state()

        env.sim.viewer_init()
        render_state = env.sim.viewer_render()
        while render_state:
            start_time = time.time()
            if not env.sim.viewer_paused():
                state = torch.Tensor(state).float()
                if actor is None:
                    action = np.random.uniform(-0.2, 0.2, env.action_size)
                else:
                    action = actor(state).numpy()
                state, reward, done, _ = env.step(action)
                episode_length += 1
                episode_reward.append(reward)
            render_state = env.sim.viewer_render()
            delaytime = max(0, env.default_policy_rate/2000 - (time.time() - start_time))
            time.sleep(delaytime)
            if episode_length == episode_length_max or done:
                print(f"Episode length = {episode_length}, Average reward is {np.mean(episode_reward)}.")
                state = env.reset()
                episode_length = 0
                if hasattr(actor, 'init_hidden_state'):
                    actor.init_hidden_state()

def interactive_eval(actor, env):
    """Simply evaluating policy in visualization window with user input

    Args:
        actor: Actor loaded outside this function. If Actor is None, this function will evaluate
            noisy actions without any policy.
        env: Environment instance for actor
        episode_length_max (int, optional): Max length of episode for evaluation. Defaults to 500.
    """
    if actor is None:
        raise RuntimeError("Interactive eval requires a non-null actor network for eval")
    keyboard = Keyboard()
    print('\033[92m' + "Feeding keyboard inputs to policy for interactive eval mode.")
    print("Type commands into the terminal window to avoid interacting with the mujoco viewer keybinds." + '\033[0m')
    with torch.no_grad():
        state = env.reset()
        done = False
        episode_length = 0
        episode_reward = []

        if hasattr(actor, 'init_hidden_state'):
            actor.init_hidden_state()

        env.sim.viewer_init()
        render_state = env.sim.viewer_render()
        env.display_controls_menu()
        while render_state:
            start_time = time.time()
            cmd = keyboard.get_input()
            if not env.sim.viewer_paused():
                state = torch.Tensor(state).float()
                action = actor(state).numpy()
                if cmd is not None:
                    env.interactive_control(cmd)
                state, reward, done, _ = env.step(action)
                episode_length += 1
                episode_reward.append(reward)
            render_state = env.sim.viewer_render()
            delaytime = max(0, env.default_policy_rate/2000 - (time.time() - start_time))
            time.sleep(delaytime)
        # clear terminal on ctrl+q
        os.system('cls||clear')
def eval_no_vis(actor, env, episode_length_max=300):
    """Simply evaluating policy without visualization

    Args:
        actor: Actor loaded outside this function. If Actor is None, this function will evaluate
            noisy actions without any policy.
        env: Environment instance for actor
        episode_length_max (int, optional): Max length of episode for evaluation. Defaults to 500.
    """
    with torch.no_grad():
        state = env.reset()
        done = False
        episode_length = 0
        episode_reward = []

        if hasattr(actor, 'init_hidden_state'):
            actor.init_hidden_state()

        while True:
            state = torch.Tensor(state).float()
            if actor is None:
                action = np.random.uniform(-0.2, 0.2, env.action_size)
            else:
                action = actor(state).numpy()
            state, reward, done, _ = env.step(action)
            episode_length += 1
            episode_reward.append(reward)
            if episode_length == episode_length_max or done:
                print(f"Episode length = {episode_length}, Average reward is {np.mean(episode_reward)}.")
                state = env.reset()
                episode_length = 0
                if hasattr(actor, 'init_hidden_state'):
                    actor.init_hidden_state()
