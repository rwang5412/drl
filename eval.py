import torch
import argparse

from util.eval_utils import simple_eval

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
    args = parser.parse_args()

    from nn.actor import LSTMActor, FFActor
    actor = LSTMActor(input_dim=42, action_dim=10, layers=[128,128], bounded=False, learn_std=False, std=0.1)
    actor_state_dict = torch.load('./pretrained_models/speed_locomotion_vonmises_clock.pt', map_location='cpu')
    actor.load_state_dict(actor_state_dict)
    simple_eval(actor=actor, args=args)
