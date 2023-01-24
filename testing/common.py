# Testing utils for Digit ar-control
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

# Motor set for testing for Digit ar-control
MOTOR_POSITION_SET={}
MOTOR_POSITION_SET['pos1'] = [
-0.0462933, -0.0265814, 0.19299, -0.3, -0.0235182, -0.0571617,
0.0462933, 0.0265814, -0.19299, 0.3, -0.0235182, 0.0571617,
-0.3, 0.943845, 0.0, 0.3633,
0.3, -0.943845, 0.0, -0.3633,
]

MOTOR_POSITION_SET['pos2'] = [
0.332, -0.0265814, 0.19299, 0.218647, 0.0235182, -0.0571617,
-0.332, 0.0265814, -0.19299, -0.218647, -0.0235182, 0.0571617,
-0.106437, 0.89488, -0.00867663, 0.344684, 
0.106339, -0.894918, 0.00889888, -0.344627
]

MOTOR_POSITION_SET['pos3'] = [
-0.0462933, -0.5265814, 0.19299, -0.3, -0.0235182, -0.0571617,
0.0462933, 0.5265814, -0.19299, 0.3, -0.0235182, 0.0571617,
-0.3, 0.0943845, 0.0,-0.3633,
0.3, -0.0943845, 0.0, 0.3633,
]

# Readable strings for motor and joint definitions. These are also used for testing.
DIGIT_MOTOR_NAME = [
  'left-leg/hip-roll', 'left-leg/hip-yaw', 'left-leg/hip-pitch', 'left-leg/knee', 
  'left-leg/toe-a', 'left-leg/toe-b',
  'left-arm/shoulder-roll','left-arm/shoulder-pitch', 'left-arm/shoulder-yaw', 'left-arm/elbow',
  'right-leg/hip-roll', 'right-leg/hip-yaw', 'right-leg/hip-pitch', 'right-leg/knee', 
  'right-leg/toe-a', 'right-leg/toe-b',
  'right-arm/shoulder-roll','right-arm/shoulder-pitch', 'right-arm/shoulder-yaw', 'right-arm/elbow'
]

DIGIT_JOINT_NAME = [
  'left-leg/shin', 'left-leg/tarsus', 'left-leg/heel-spring', 
  'left-leg/toe-pitch', 'left-leg/toe-roll',
  'right-leg/shin', 'right-leg/tarsus', 'right-leg/heel-spring', 
  'right-leg/toe-pitch', 'right-leg/toe-roll'
]

CASSIE_MOTOR_NAME = [
  "left-hip-roll", "left-hip-yaw", "left-hip-pitch", "left-knee", "left-foot", 
  "right-hip-roll", "right-hip-yaw", "right-hip-pitch", "right-knee", "right-foot"
]

CASSIE_JOINT_NAME = ["left-shin", "left-tarsus", "right-shin", "right-tarsus"]