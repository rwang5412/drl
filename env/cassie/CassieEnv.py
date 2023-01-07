# Handles all 2kHz stuff
class CassieEnv(RobotEnv):

    def __init__(simulator_type: str, clock: bool):

        if simulator_type == "mujoco":
            self.sim = MjCassieSim()
        elif simulator_type == 'libcassie':
            self.sim = CassieSim()

    def _sim_forward(self, action_type, cmd):
        # Handle simulator control input to step
        # runs at 2Khz or the simulator rate
        #TODO: ceate struct for each cmd type
        if action_type == "torque":
            u = motor_model(cmd)
        elif action_type == "pd":
            u = pd(cmd)
        elif action_type == "task":
            u = ID_controller(cmd)
        self.sim.step(u)

        # Update sim trackers (signals higher than policy rate, like GRF, etc)
        self.foot_GRF = blah
        # For MjCassieSim, internally will call 
        # mj.step(self.mjModel, self.mjData)