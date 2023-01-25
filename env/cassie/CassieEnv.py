from env import GenericEnv
from sim import MjCassieSim, LibCassieSim

# Handles all 2kHz stuff
class CassieEnv(GenericEnv):
    def __init__(self, simulator_type: str, clock: bool):
        """Template class for Cassie with common functions (templates).

        Args:
            simulator_type (str): "mujoco" or "libcassie"
            clock (bool): "linear" or "von-Mises" or None
        """
        super().__init__()
        if simulator_type == "mujoco":
            self.sim = MjCassieSim()
        elif simulator_type == 'libcassie':
            self.sim = LibCassieSim()
        else:
            raise RuntimeError(f"Simulator type {simulator_type} not correct!")

        self.clock = clock

    def reset(self):
        raise NotImplementedError

    def step(self, action:list):
        """Step simulator by env's frequency. Still accesses signals at simulator rate (2kHz).
        User should add any 2kHz signals here and fetch it inside each specific env.

        Args:
            action (list): _description_
        """
        for _ in range(10):
            self.sim.set_PD(P_gain=action, D_targ=0, P_targ=0, D_gain=0)
            self.sim.sim_forward(action)

            # Update sim trackers (signals higher than policy rate, like GRF, etc)
            self.foot_GRF = blah
            # For MjCassieSim, internally will call 
            # mj.step(self.mjModel, self.mjData)
