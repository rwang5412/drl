import importlib

import numpy as np

# from rewards.blah import foo_reward
# from env.util import Clock
from env import DigitEnv

class DigitEnvClock(DigitEnv):

    def __init__(self,
                 clock: str,
                 reward_name: str,
                 simulator_type: str,
                 terrain: bool,
                 policy_rate: int,
                 dynamics_randomization: bool):
        
        super().__init__(simulator_type=simulator_type,
                         terrain=terrain,
                         policy_rate=policy_rate,
                         dynamics_randomization=dynamics_randomization)

        # Define env specifics
        self.clock = clock
        self.observation_space = None
        self.action_space = None

        # Load reward module
        # self.reward = importlib.import_module(name='env.rewards.'+reward_name)
        # self.w = setup_reward_components(self, incentive=self.incentive)
        # self.compute_reward = self.reward.compute_reward
        # self.compute_done = self.reward.compute_done

    def reset(self):
        """Reset simulator and env variables.

        Returns:
            state (np.ndarray): the s in (s, a, s')
        """
        self.reset_simulation()
        self.traj_idx = 0
        self.orient_add = 0
        return self.get_state()

    def step(self, action: np.ndarray):

        if self.dynamics_randomization:
            self.policy_rate = self.default_policy_rate + np.random.randint(-5, 6)
        else:
            self.policy_rate = self.default_policy_rate

        # Step simulation by n steps. This call will update self.tracker_fn.
        simulator_repeat_steps = int(self.sim.simulator_rate / self.policy_rate)
        self.step_simulation(action, simulator_repeat_steps)
        # Reward for taking current action before changing quantities for new state
        r = self.compute_reward(action)

        self.traj_idx += 1
        self.last_action = action

        return self.get_state(), r, self.compute_done(), {}

    def compute_reward(self, action: np.ndarray):
        return 1

    def compute_done(self):
        pass

    def get_state(self):
        return self.get_robot_state()

    def get_action_mirror_indices(self):
        raise NotImplementedError

    def get_state_mirror_indices(self):
        raise NotImplementedError
