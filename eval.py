import torch
import time
import argparse

import numpy as np

from util.env_factory import env_factory

def simple_eval(actor, run_args, traj_len_max=1000):
    env = env_factory(**vars(run_args))()
    
    with torch.no_grad():
        state = env.reset()
        done = False
        traj_len = 0
        
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
                traj_len += 1
            render_state = env.sim.viewer_render()
            delaytime = max(0, 50/2000 - (time.time() - start_time))
            time.sleep(delaytime)
            if traj_len == traj_len_max or done:
                state = env.reset()
                traj_len = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """Environment specific kwargs
    """
    parser.add_argument('--env-name', default="CassieEnvClock")
    parser.add_argument('--simulator-type', default="mujoco")
    parser.add_argument('--clock-type', default="von_mises")
    parser.add_argument('--reward-name', default="locomotion_vonmises_clock_reward")
    parser.add_argument('--policy-rate', default=40)
    parser.add_argument('--dynamics-randomization', default=False)
    parser.add_argument('--terrain', default=False)
    
    """Actor/Critic specific kwargs
    """
    # parser.add_argument('--input-dim', type=int, default=1)
    # parser.add_argument('--action-dim', type=int, default=1)
    # parser.add_argument('--layers', nargs='+', type=int, default=[128,128])
    # parser.add_argument('--learn-std', default=False)
    # parser.add_argument('--std', default=1)
    
    # Process args
    args = parser.parse_args()
    # print(args)
    
    from nn.actor import LSTMActor, FFActor
    actor = LSTMActor(input_dim=42, action_dim=10, layers=[128,128], bounded=False, learn_std=False, std=0.1)
    actor_state_dict = torch.load('./pretrained_models/speed_locomotion_vonmises_clock.pt', map_location='cpu')
    actor.load_state_dict(actor_state_dict)
    simple_eval(actor=actor, run_args=args)
