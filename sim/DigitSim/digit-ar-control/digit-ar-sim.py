import asyncio
import agility
import agility.messages as msg
import atexit

from llapi.llapictypes import (
    llapi_command_t, 
    llapi_observation_t, 
    NUM_MOTORS, 
    Damping, 
    Locomotion
)

from llapi.llapictypes import (
    llapi_init, 
    llapi_init_custom, 
    llapi_get_limits, 
    llapi_get_observation, 
    llapi_send_command,
    llapi_connected,
    llapi_get_error_shutdown_delay
)

class ArControl:
    """
    This class will handle simulator initialization and close. The path to ar-control binary file 
    should be provided at construction. Conf files (args) can be loaded as well.
    @TODO (helei): best way to sync up ports/ips for sim/api/llapi?
    """
    def __init__(self, 
                 args,
                 path_to_arcontrol: str,
                 address: str = '127.0.0.1',
                 port: int = 8080,
                 connect_timeout: float = 1.0):
        # Initialize simulator and ar-control stack
        self.sim = agility.Simulator(path_to_arcontrol, *args)
        # Initialize API to comm with simulator/ar-control
        self.api = agility.JsonApi(address=address,
                                   port=port,
                                   connect_timeout=connect_timeout)
        # Initialize LLAPI to intercept commands and update obs
        llapi_init(address)
        while not llapi_get_observation(llapi_observation_t()):
            llapi_send_command(llapi_command_t())
        print("Connected to LLAPI!")
        
        # Register atexit calls
        # TODO: helei, how to deal with properly async atexit
        atexit.register(self.simulator_close)
        # atexit.register(self.api_close)
        
        # Other parameters
        self.simulator_frequency = 2000 #Hz. This is really up to 400Hz
        self.actuator_limit = llapi_get_limits()[0]
        self.obs = llapi_observation_t()
        self.cmd = llapi_command_t()
        self.obs_return_code = None

    # Need to call below to connect API to simulator
    async def api_setup(self):
        """Connect with Simulator, reset simulator, and setup API priviledges.

        Raises:
            Exception: When connection failed, check the address and port in constructor
        """
        try:
            await self.api.connect()
        except:
            raise Exception("Cannot connect with simulator address at ", )
        await self.api.request_privilege('change-action-command')
        await self.api.send('simulator-reset')
        # A bit wait for simulator settles
        await asyncio.sleep(0.5)
        # Pause simulator and record timestamp
        await self.api.send(msg.SimulatorPause())
        self.t = (await ar.api.query(msg.GetTimestamp()))['run-time']

    async def sim_forward(self, dt:float, actions:list=None):
        """Step simulator by dt and update/track self.t.
        NOTE: Tries to run step at 2kHz, but tested can only run up to 200Hz, 
        so max dt that correctly sim forward is 0.005s. Keep 2kHz does not affect accuracy.

        Args:
            dt (float): Amount of time to forward simulator
            actions (list[float]): A list of actions from upstream envs
        """
        self.t = (await ar.api.query(msg.GetTimestamp()))['run-time']
        time_start = self.t
        while dt >= self.t - time_start:
            await self.api.send(msg.SimulatorStep(dt=1/self.simulator_frequency))
            self.t = (await ar.api.query(msg.GetTimestamp()))['run-time']
            # Update obs and cmd in simulator rate
            # Because LLAPI gets from UDP, obs update has to be in a fast loop in order to sync up
            # TODO: helei, see how this part fits with GeneriSim, set_action() etc
            await ar.api.send(msg.SimulatorPause())
            self.get_observations()
            self.send_commands(actions=actions)
        # print(self.t - time_start)

    def get_observations(self):
        """Update the observations from LLAPI
        """
        # TODO: helei, figure out how to deal with return code to ensure this gets updated msg 
        # from LLAPI
        self.obs_return_code = llapi_get_observation(ar.obs)

    def send_commands(self, actions:list=None):
        """Send commands to LLAPI

        Args:
            actions (list): Coming from upstream env side
        """
        # TODO: helei, add PD setpoint into LLAPI header so the torques can be updated in c level
        if actions is not None: # Write to cmd struct if actions present
            apply_command = True
            for i in range(NUM_MOTORS):
                ar.cmd.motors[i].torque = 150*(actions[i] - self.obs.motor.position[i])
                ar.cmd.motors[i].velocity = 0.0
                ar.cmd.motors[i].damping = 0.75 * self.actuator_limit.damping_limit[i]
        else:
            apply_command = False
        ar.cmd.fallback_opmode = Locomotion
        ar.cmd.apply_command = apply_command
        llapi_send_command(ar.cmd)
    
    async def api_close(self):
        await self.api.close()

    def simulator_close(self):
        self.sim.close()


"""
The following is more like a test file that test the ar-control above.
We can also fit below inside arcontrol class or into GenericSim or exact env
"""

sim_path = './ar-control'
conf = []
conf += ["""
# Add robot to world
[[model-list]]
model = "robot"
pose = {xyz = [0, 0, 1.0]}

[simulator]
free-run=true

[planning]
initial-operation-mode = "locomotion"

[lowlevelapi]
enable = true
# Time in seconds between Low-level API packets before control program
# disables robot
timeout = 0.05
# Port for incoming and outgoing Low-level API communications.
# Change if these settings interfere with any other programs or network
# Traffic on the payload computer. Must match settings in lowlevelapi.c
# (lines 181 and 183). These should not be changed under normal operation
listen-port = 25500
send-port = 25501
"""]

ar = ArControl(path_to_arcontrol=sim_path, args=conf)

motor_pos_set={}
motor_pos_set['pos1'] = [
-0.0462933,
-0.0265814,
0.19299,
-0.3,
-0.0235182,
-0.0571617,
0.0462933,
0.0265814,
-0.19299,
0.3,
-0.0235182,
0.0571617,
-0.3,
0.943845,
0.0,
0.3633,
0.3,
-0.943845,
0.0,
-0.3633,
]

motor_pos_set['pos2'] = [
0.332,
-0.0265814,
0.19299,
0.218647,
0.0235182,
-0.0571617,
-0.332,
0.0265814,
-0.19299,
-0.218647,
-0.0235182,
0.0571617,
-0.106437,
0.89488,
-0.00867663,
0.344684,
0.106339,
-0.894918,
0.00889888,
-0.344627
]

motor_pos_set['pos3'] = [
-0.0462933,
-0.5265814,
0.19299,
-0.3,
-0.0235182,
-0.0571617,
0.0462933,
0.5265814,
-0.19299,
0.3,
-0.0235182,
0.0571617,
-0.3,
0.0943845,
0.0,
-0.3633,
0.3,
-0.0943845,
0.0,
0.3633,
]

async def test():
    # Need to call this to connect API with simulator
    await ar.api_setup()
    # Send API actions, but this won;t move robot, since the followings mannually sim-step
    await ar.api.send(msg.ActionMove(velocity={'rpyxyz':[0,0,0,0.5,0,0]}))
    
    # Test sim forward while running ar-control
    for i in range(2):
        await ar.sim_forward(dt=1)
        print("time={:3.2f}, x={:3.1f}, llapi-connected={}".format(
            ar.t, ar.obs.base.translation[0], llapi_connected()))
    
    # Test LLAPI mode toggle
    # To fully enable LLAPI and take over ar-control. Call sim_forward with actions for step 1 & 2
    # and API call for step 3.
    # 1. Write/Send commands struct
    # 2. Toggle apply_command to be True
    # 3. Enable LLAPI mode via API call
    await ar.sim_forward(dt=1, actions=motor_pos_set['pos1'])
    act = (await ar.api.query(msg.GetActionCommand()))
    if act['action'] != 'action-set-operation-mode':
        await ar.api.send(msg.ActionSetOperationMode(mode="low-level-api"))
    print("LLAPI ON")

    # Since LLAPI mode on, we can keep sending to execute custom commands.
    # Robot will fall in this test case. 
    # NOTE: still mystery about when Damping mode will turn on
    for i in range(20):
        if i%3==1:
            await ar.sim_forward(dt=1, actions=motor_pos_set['pos1'])
        elif i%3==2:
            await ar.sim_forward(dt=1, actions=motor_pos_set['pos2'])
        else:
            await ar.sim_forward(dt=1, actions=motor_pos_set['pos3'])
        print("time={: 4.2f}, z-vel={: 3.2f}, llapi-connect={}, obs-error={}".format(
            ar.t, ar.obs.base.linear_velocity[2], llapi_connected(), ar.obs.error))

        # Example of applying damping mode
        # TODO: helei, ideally this will be triggered auto by ar-control. Need to try with hardware.
        # if ar.obs.imu.linear_acceleration[2] > 0.6:
        #     await ar.api.send(msg.ActionSetOperationMode(mode="damping"))
        #     break
        # await asyncio.sleep(5)

    await ar.api_close()
    ar.simulator_close()
    
asyncio.run(test())
