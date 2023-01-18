from sim.digit_sim import DigitArSim
import asyncio
import agility
import agility.messages as msg
import atexit
import asyncio_atexit
import numpy as np
import time

from sim.digit_sim.digit_ar_sim.llapi.llapictypes import (
    llapi_connected,
)

"""
The following is more like a test file that test the ar-control above.
We can also fit below inside arcontrol class or into GenericSim or exact env
"""

sim_path = './sim/digit_sim/digit_ar_sim/ar-control'
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

async def test_ar_connect():
    ar = DigitArSim(path_to_ar_control=sim_path, args=conf)
    # Need to call this to connect API with simulator
    await ar.api_setup()
    print("connected with sim/api/llapi")
    await ar.api.send(msg.SimulatorStart())
    await ar.api_close()
    ar.simulator_close()

async def test_ar_api_goto():
    ar = DigitArSim(path_to_ar_control=sim_path, args=conf)
    # Need to call this to connect API with simulator
    await ar.api_setup()
    await ar.api.send(msg.SimulatorStart())
    ret = (await ar.api.wait_action(msg.ActionGoto(target={"xy": [2, 0]})))
    print(ret)
    await ar.api_close()
    ar.simulator_close()
    
async def test_ar_sim_forward():
    ar = DigitArSim(path_to_ar_control=sim_path, args=conf)
    # Need to call this to connect API with simulator
    await ar.api_setup()
    # Send API actions, but this won't move robot, since the followings mannually sim-step
    await ar.api.send(msg.ActionGoto(target={"xy": [1, 0]}))
    
    # Test sim forward while running ar-control
    for _ in range(10):
        await ar.sim_forward(dt=0.5)
        await ar.api.send(msg.SimulatorPause())
        await asyncio.sleep(0.1)
        print("time={:3.2f}, x={:3.1f}, llapi-connected={}".format(
            ar.t, ar.obs.base.translation[0], llapi_connected()))
    if np.abs(ar.obs.base.translation[0] - 1) < 1e-2:
        print("success")
    await ar.api_close()
    ar.simulator_close()
    
async def test_ar_sim_llapi():
    ar = DigitArSim(path_to_ar_control=sim_path, args=conf)
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
    for i in range(2):
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
