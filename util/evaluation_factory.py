import numpy as np
import sys
import termios
import time
import torch

from util.keyboard import Keyboard
from util.colors import OKGREEN, FAIL

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
                done = False
                state = env.reset()
                episode_length = 0
                if hasattr(actor, 'init_hidden_state'):
                    actor.init_hidden_state()
                # Seems like Mujoco only allows a single mjContext(), and it prefers one context
                # with one window when modifying mjModel. So for onscreen dual window, we re-init
                # the non-main window, ie egocentric view here.
                if hasattr(env.sim, 'renderer'):
                    if env.sim.renderer is not None:
                        print("re-init non-primary screen renderer")
                        env.sim.renderer.close()
                        env.sim.init_renderer(offscreen=env.offscreen,
                                              width=env.depth_image_dim[0], height=env.depth_image_dim[1])

def interactive_eval(actor, env, episode_length_max=300, critic=None):
    """Simply evaluating policy in visualization window with user input

    Args:
        actor: Actor loaded outside this function. If Actor is None, this function will evaluate
            noisy actions without any policy.
        env: Environment instance for actor
        episode_length_max (int, optional): Max length of episode for evaluation. Defaults to 500.
    """
    if actor is None:
        raise RuntimeError(F"{FAIL}Interactive eval requires a non-null actor network for eval")

    keyboard = Keyboard()
    print(f"{OKGREEN}Feeding keyboard inputs to policy for interactive eval mode.")
    print("Type commands into the terminal window to avoid interacting with the mujoco viewer keybinds." + '\033[0m')
    with torch.no_grad():
        state = env.reset(interactive_evaluation=True)
        done = False
        episode_length = 0
        episode_reward = []

        if hasattr(actor, 'init_hidden_state'):
            actor.init_hidden_state()
        if hasattr(critic, 'init_hidden_state'):
            critic.init_hidden_state()

        env.sim.viewer_init()
        render_state = env.sim.viewer_render()
        env.display_controls_menu()
        env.display_control_commands()
        while render_state:
            start_time = time.time()
            cmd = keyboard.get_input()
            if not env.sim.viewer_paused():
                state = torch.Tensor(state).float()
                action = actor(state).numpy()
                state, reward, done, _ = env.step(action)
                episode_length += 1
                episode_reward.append(reward)
                if critic is not None:
                    if hasattr(env, 'get_privilege_state'):
                        critic_state = env.get_privilege_state()
                    else:
                        critic_state = state
                    # print(f"Critic value = {critic(torch.Tensor(critic_state)).numpy() if critic is not None else 'N/A'}")
            if cmd is not None:
                env.interactive_control(cmd)
            if cmd == "quit":
                done = True
            if cmd == "menu":
                env.display_control_commands(erase=True)
                env.display_controls_menu()
                env.display_control_commands()
            render_state = env.sim.viewer_render()
            delaytime = max(0, env.default_policy_rate/2000 - (time.time() - start_time))
            time.sleep(delaytime)
            if done:
                state = env.reset(interactive_evaluation=True)
                env.display_control_commands(erase=True)
                print(f"Episode length = {episode_length}, Average reward is {np.mean(episode_reward)}.")
                env.display_control_commands()
                episode_length = 0
                if hasattr(actor, 'init_hidden_state'):
                    actor.init_hidden_state()

        # clear terminal on ctrl+q
        print(f"\033[{env.num_menu_backspace_lines - 1}B\033[K")
        termios.tcdrain(sys.stdout)
        time.sleep(0.1)
        termios.tcflush(sys.stdout, termios.TCIOFLUSH)

def simple_eval_offscreen(actor, env, episode_length_max=300):
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
