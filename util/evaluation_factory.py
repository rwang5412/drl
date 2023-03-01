import torch
import time
import argparse

import numpy as np

from util.env_factory import env_factory

def simple_eval(actor, env_name, env_args, episode_length_max=300):
    """Simply evaluating policy without UI via terminal

    Args:
        actor: Actor loaded outside this function. If Actor is None, this function will evaluate
            noisy actions without any policy.
        args: Arguments for environment.
        episode_length_max (int, optional): Max length of episode for evaluation. Defaults to 500.
    """
    # Load environment
    env = env_factory(env_name, env_args)()

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
