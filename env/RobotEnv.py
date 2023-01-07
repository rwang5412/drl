
class RobotEnv:

    """
    Define generic environment functions that are needed for RL. Should define (not implement) all of the functions that ppo.py sampling uses.
    """
    
    def __init__():
        self.sim = None
        pass

    def step(self, action):
        raise NotImplementedError
    
    def get_state(self):
        raise NotImplementedError

    def compute_reward(self):
        raise NotImplementedError

    def set_action(self):
        raise NotImplementedError

    def _sim_forward
