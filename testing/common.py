SIM_PATH = './sim/digit_sim/digit_ar_sim/ar-control'

ROBOT_CONFIG = ["""
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

MOTOR_POSITION_SET={}
MOTOR_POSITION_SET['pos1'] = [
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

MOTOR_POSITION_SET['pos2'] = [
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

MOTOR_POSITION_SET['pos3'] = [
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
