import torch
import time
import argparse

import numpy as np

from util.env_factory import env_factory

def simple_eval(actor, args, episode_length_max=300):
    # Load environment
    env = env_factory(**vars(args))()

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

if __name__ == "__main__":

    """
    These parsers will be removed and replaced by loading args from dict.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="CassieEnvClock")
    parser.add_argument('--simulator-type', default="mujoco")
    parser.add_argument('--clock-type', default="von_mises")
    parser.add_argument('--reward-name', default="locomotion_vonmises_clock_reward")
    parser.add_argument('--policy-rate', default=40)
    parser.add_argument('--dynamics-randomization', default=False)
    parser.add_argument('--terrain', default=False)

    # Process args
    args = parser.parse_args()

    from nn.actor import LSTMActor, FFActor
    actor = LSTMActor(input_dim=42, action_dim=10, layers=[128,128], bounded=False, learn_std=False, std=0.1)
    actor_state_dict = torch.load('./pretrained_models/speed_locomotion_vonmises_clock.pt', map_location='cpu')
    actor.load_state_dict(actor_state_dict)
    simple_eval(actor=actor, args=args)
