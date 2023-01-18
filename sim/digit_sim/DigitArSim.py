import asyncio
import agility
import agility.messages as msg
import atexit
import asyncio_atexit
import numpy as np
import time

from .digit_ar_sim.llapi.llapictypes import (
    llapi_command_t, 
    llapi_observation_t, 
    NUM_MOTORS, 
    NUM_JOINTS,
    Damping, 
    Locomotion
)

from .digit_ar_sim.llapi.llapictypes import (
    llapi_init, 
    llapi_init_custom, 
    llapi_get_limits, 
    llapi_get_observation, 
    llapi_send_command,
    llapi_connected,
    llapi_get_error_shutdown_delay
)

class DigitArSim:
    """
    This class will handle simulator initialization and close. The path to ar-control binary file 
    should be provided at construction. Conf files (args) can be loaded as well.
    @TODO (helei): best way to sync up ports/ips for sim/api/llapi?
    """
    def __init__(self, 
                 args,
                 path_to_ar_control: str,
                 address: str = '127.0.0.1',
                 port: int = 8080,
                 connect_timeout: float = 1.0):
        self.address = address
        # Initialize simulator and ar-control stack
        self.sim = agility.Simulator(path_to_ar_control, *args)
        # Initialize API to comm with simulator/ar-control
        self.api = agility.JsonApi(address=address,
                                   port=self.port,
                                   connect_timeout=connect_timeout)
        # Initialize LLAPI to intercept commands and update obs
        llapi_init(address)
        start = time.monotonic()
        while not llapi_get_observation(llapi_observation_t()):
            llapi_send_command(llapi_command_t())
            if time.monotonic() - start > 2:
                raise RuntimeError("Cannot connect to LLAPI! Check port and IP address setup!")
        
        # Register atexit calls
        # TODO: helei, how to deal with properly async atexit
        atexit.register(self.simulator_close)
        # asyncio_atexit.register(self.api_close)
        
        # Other parameters
        self.simulator_frequency = 2000 #Hz. This is really up to 400Hz
        self.actuator_limit = llapi_get_limits()[0]
        self.obs = llapi_observation_t()
        self.cmd = llapi_command_t()
        self.obs_return_code = None

    async def api_setup(self):
        """Connect with Simulator, reset simulator, and setup API priviledges. Must call this.

        Raises:
            Exception: When connection failed, check the address and port in constructor.
        """
        try:
            await self.api.connect()
        except:
            raise Exception("Cannot connect api with ip at ", self.address, " port at ", self.port)
        await self.api.request_privilege('change-action-command')
        await self.api.send('simulator-reset')
        # A bit wait for simulator settles
        await asyncio.sleep(0.5)
        # Pause simulator and record timestamp
        await self.api.send(msg.SimulatorPause())
        self.t = (await self.api.query(msg.GetTimestamp()))['run-time']

    async def sim_forward(self, dt:float, actions:list=None):
        """Step simulator by dt and update/track self.t.
        NOTE: Tries to run step at 2kHz, but tested can only run up to 200Hz, 
        so max dt that correctly sim forward is 0.005s. Keep 2kHz does not affect accuracy.

        Args:
            dt (float): Amount of time to forward simulator
            actions (list[float]): A list of actions from upstream envs
        """
        self.t = (await self.api.query(msg.GetTimestamp()))['run-time']
        time_start = self.t
        while dt >= self.t - time_start:
            await self.api.send(msg.SimulatorStep(dt=1/self.simulator_frequency))
            self.t = (await self.api.query(msg.GetTimestamp()))['run-time']
            # Update obs and cmd in simulator rate
            # Because LLAPI gets from UDP, obs update has to be in a fast loop in order to sync up
            # TODO: helei, see how this part fits with GeneriSim, set_action() etc
            await self.api.send(msg.SimulatorPause())
            self.get_observations()
            self.send_commands(actions=actions)

    def get_observations(self):
        """Update the observations from LLAPI
        """
        # TODO: helei, figure out how to deal with return code to ensure this gets updated msg 
        # from LLAPI. Since UDP-based comm and loop-rate bottleneck, obs will not be as close to 2kHz
        # signals, and in fact, we can probabaly never get 2kHz state since UDP.
        self.obs_return_code = llapi_get_observation(self.obs)

    def send_commands(self, actions:list=None):
        """Send commands to LLAPI

        Args:
            actions (list): Coming from upstream env side
        """
        # TODO: helei, add PD setpoint into LLAPI header so the torques can be updated in c level
        if actions is not None: # Write to cmd struct if actions present
            apply_command = True
            for i in range(NUM_MOTORS):
                self.cmd.motors[i].torque = 150*(actions[i] - self.obs.motor.position[i])
                self.cmd.motors[i].velocity = 0.0
                self.cmd.motors[i].damping = 0.75 * self.actuator_limit.damping_limit[i]
        else:
            apply_command = False
        self.cmd.fallback_opmode = Locomotion
        self.cmd.apply_command = apply_command
        llapi_send_command(self.cmd)
    
    """Followings are getter functions for observations.
    """
    def get_motor_position(self):
        return np.array(self.obs.motor.position[:NUM_MOTORS])
    
    def get_motor_velocity(self):
        return np.array(self.obs.motor.velocity[:NUM_MOTORS])

    def get_motor_torque(self):
        return np.array(self.obs.motor.torque[:NUM_MOTORS])
    
    def get_joint_position(self):
        return np.array(self.obs.joint.position[:NUM_JOINTS])
    
    def get_joint_velocity(self):
        return np.array(self.obs.joint.velocity[:NUM_JOINTS])
    
    def get_base_translation(self):
        return np.array(self.obs.base.translation[:3])
    
    def get_base_orientation(self):
        return np.array(self.obs.base.orientation[:4])
    
    def get_base_linear_velocity(self):
        return np.array(self.obs.base.linear_velocity[:3])
    
    def get_base_angular_velocity(self):
        return np.array(self.obs.base.angular_velocity[:3])
    
    def get_imu_linear_accleration(self):
        return np.array(self.obs.imu.linear_acceleration[:3])

    def get_imu_angular_velocity(self):
        return np.array(self.obs.imu.angular_velocity[:3])
    
    def get_imu_orientation(self):
        return np.array(self.obs.imu.orientation[:4])
    
    def viewer_init(self):
        """Placeholder since ar-control visualize via websockets automatically
        """
        pass
    
    def viewer_render(self):
        """Placeholder since ar-control visualize via websockets automatically
        """
        pass
    
    async def api_close(self):
        await self.api.close()

    def simulator_close(self):
        self.sim.close()


# """
# The following is more like a test file that test the ar-control above.
# We can also fit below inside arcontrol class or into GenericSim or exact env
# """

# sim_path = './digit_ar_sim/ar-control'
# conf = []
# conf += ["""
# # Add robot to world
# [[model-list]]
# model = "robot"
# pose = {xyz = [0, 0, 1.0]}

# [simulator]
# free-run=true

# [planning]
# initial-operation-mode = "locomotion"

# [lowlevelapi]
# enable = true
# # Time in seconds between Low-level API packets before control program
# # disables robot
# timeout = 0.05
# # Port for incoming and outgoing Low-level API communications.
# # Change if these settings interfere with any other programs or network
# # Traffic on the payload computer. Must match settings in lowlevelapi.c
# # (lines 181 and 183). These should not be changed under normal operation
# listen-port = 25500
# send-port = 25501
# """]

# ar = DigitArSim(path_to_ar_control=sim_path, args=conf)

# motor_pos_set={}
# motor_pos_set['pos1'] = [
# -0.0462933,
# -0.0265814,
# 0.19299,
# -0.3,
# -0.0235182,
# -0.0571617,
# 0.0462933,
# 0.0265814,
# -0.19299,
# 0.3,
# -0.0235182,
# 0.0571617,
# -0.3,
# 0.943845,
# 0.0,
# 0.3633,
# 0.3,
# -0.943845,
# 0.0,
# -0.3633,
# ]

# motor_pos_set['pos2'] = [
# 0.332,
# -0.0265814,
# 0.19299,
# 0.218647,
# 0.0235182,
# -0.0571617,
# -0.332,
# 0.0265814,
# -0.19299,
# -0.218647,
# -0.0235182,
# 0.0571617,
# -0.106437,
# 0.89488,
# -0.00867663,
# 0.344684,
# 0.106339,
# -0.894918,
# 0.00889888,
# -0.344627
# ]

# motor_pos_set['pos3'] = [
# -0.0462933,
# -0.5265814,
# 0.19299,
# -0.3,
# -0.0235182,
# -0.0571617,
# 0.0462933,
# 0.5265814,
# -0.19299,
# 0.3,
# -0.0235182,
# 0.0571617,
# -0.3,
# 0.0943845,
# 0.0,
# -0.3633,
# 0.3,
# -0.0943845,
# 0.0,
# 0.3633,
# ]

# async def test():
#     # Need to call this to connect API with simulator
#     await ar.api_setup()
#     # Send API actions, but this won;t move robot, since the followings mannually sim-step
#     await ar.api.send(msg.ActionMove(velocity={'rpyxyz':[0,0,0,0.5,0,0]}))
    
#     # Test sim forward while running ar-control
#     for i in range(2):
#         await ar.sim_forward(dt=1)
#         print("time={:3.2f}, x={:3.1f}, llapi-connected={}".format(
#             ar.t, ar.obs.base.translation[0], llapi_connected()))
    
#     # Test LLAPI mode toggle
#     # To fully enable LLAPI and take over ar-control. Call sim_forward with actions for step 1 & 2
#     # and API call for step 3.
#     # 1. Write/Send commands struct
#     # 2. Toggle apply_command to be True
#     # 3. Enable LLAPI mode via API call
#     await ar.sim_forward(dt=1, actions=motor_pos_set['pos1'])
#     act = (await ar.api.query(msg.GetActionCommand()))
#     if act['action'] != 'action-set-operation-mode':
#         await ar.api.send(msg.ActionSetOperationMode(mode="low-level-api"))
#     print("LLAPI ON")

#     # Since LLAPI mode on, we can keep sending to execute custom commands.
#     # Robot will fall in this test case. 
#     # NOTE: still mystery about when Damping mode will turn on
#     for i in range(2):
#         if i%3==1:
#             await ar.sim_forward(dt=1, actions=motor_pos_set['pos1'])
#         elif i%3==2:
#             await ar.sim_forward(dt=1, actions=motor_pos_set['pos2'])
#         else:
#             await ar.sim_forward(dt=1, actions=motor_pos_set['pos3'])
#         print("time={: 4.2f}, z-vel={: 3.2f}, llapi-connect={}, obs-error={}".format(
#             ar.t, ar.obs.base.linear_velocity[2], llapi_connected(), ar.obs.error))

#         # Example of applying damping mode
#         # TODO: helei, ideally this will be triggered auto by ar-control. Need to try with hardware.
#         # if ar.obs.imu.linear_acceleration[2] > 0.6:
#         #     await ar.api.send(msg.ActionSetOperationMode(mode="damping"))
#         #     break
#         # await asyncio.sleep(5)

#     await ar.api_close()
#     ar.simulator_close()
    
# asyncio.run(test())
