import mujoco as mj
import numpy as np
import pathlib

from sim import MujocoSim


class MjDigitSim(MujocoSim):

    """
    Wrapper for Digit Mujoco. This class only defines several specifics for Digit.
    """
    def __init__(self, model_name: str = "digit-v3-new.xml"):
        model_path = pathlib.Path(__file__).parent.resolve() / model_name
        # Number of sim steps before commanded torque is actually applied
        self.torque_delay_cycles = 6
        self.torque_efficiency = 1.0

        self.motor_position_inds = [7, 8, 9, 14, 18, 23, 30, 31, 32, 33, 34, 35, 36, 41, 45, 50, 57, 58, 59, 60]
        self.motor_velocity_inds = [6, 7, 8, 12, 16, 20, 26, 27, 28, 29, 30, 31, 32, 36, 40, 44, 50, 51, 52, 53]
        self.joint_position_inds = [15, 16, 17, 28, 29, 42, 43, 44, 55, 56]
        self.joint_velocity_inds = [13, 14, 15, 24, 25, 37, 38, 39, 48, 49]

        self.base_position_inds = [0, 1, 2]
        self.base_orientation_inds = [3, 4, 5, 6]
        self.base_linear_velocity_inds = [0, 1, 2]
        self.base_angular_velocity_inds = [3, 4, 5]
        self.base_body_name = "torso/base"
        self.feet_site_name = ["left-foot-mid", "right-foot-mid"] # pose purpose
        self.feet_body_name = ["left-leg/toe-roll", "right-leg/toe-roll"] # force purpose
        self.hand_body_name = ["left-arm/elbow", "right-arm/elbow"]

        self.num_actuators = self.model.nu
        self.num_joints = len(self.joint_position_inds)
        self.reset_qpos = np.array([0, 0, 1,  1, 0 ,0 ,0,
                        3.33020155e-01, -2.66178730e-02, 1.92369587e-01,
                        9.93409734e-01, -1.04126145e-03, 1.82534311e-03, 1.14597921e-01,
                        2.28971047e-01, 1.48527831e-03, -2.31455693e-01, 4.55857916e-04,
                        -1.29734322e-02,
                        9.89327705e-01, 1.45524756e-01, 1.73630859e-03, 7.08678995e-03,
                        -2.03852305e-02,
                        9.88035432e-01, 1.53876629e-01, 6.59769560e-05, 1.03905844e-02,
                        -3.52778547e-03, -5.54992074e-02,
                        -1.05542715e-01, 8.94852532e-01, -8.63756398e-03, 3.44780280e-01,
                        -3.33020070e-01, 2.66195360e-02, -1.92382190e-01,
                        9.93409659e-01, 1.04481446e-03, 1.82489637e-03, -1.14598546e-01,
                        -2.28971188e-01, -1.48636971e-03, 2.31454977e-01, -4.53425792e-04,
                        1.41299940e-02,
                        9.89323654e-01, -1.45521550e-01, 1.80177609e-03, -7.67726135e-03,
                        1.93767895e-02,
                        9.88041418e-01, -1.53872399e-01, 1.26073837e-05, -9.87117790e-03,
                        2.36874210e-03, 5.55559678e-02,
                        1.05444698e-01, -8.94890429e-01, 8.85979401e-03, -3.44723293e-01
                        ])

        # NOTE: Have to call super init AFTER index arrays are defined
        super().__init__(model_path=model_path)

        self.simulator_rate = int(1 / self.model.opt.timestep)

        self.offset = self.reset_qpos[self.motor_position_inds]
        self.kp = np.array([200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0,
                            200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0])
        self.kd = np.array([10.0, 10.0, 20.0, 20.0, 7.0, 7.0, 10.0, 10.0, 10.0, 10.0,
                            10.0, 10.0, 20.0, 20.0, 7.0, 7.0, 10.0, 10.0, 10.0, 10.0])




