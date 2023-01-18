import subprocess
if subprocess.run('nvidia-smi').returncode:
  raise RuntimeError(
		'Cannot communicate with GPU. '
		'Make sure you are using a GPU Colab runtime. '
		'Go to the Runtime menu and select Choose runtime type.')
print('Setting environment variable to use GPU rendering:')
# %env MUJOCO_GL=egl
import os
os.environ['MUJOCO_GL']='glx'
os.environ['PYOPENGL_PLATFORM'] = 'glx'

# try:
# 	print('Checking that the installation succeeded:')
import mujoco
# 	mujoco.MjModel.from_xml_string('<mujoco/>')
# except Exception as e:
#   raise e from RuntimeError(
# 		'Something went wrong during installation. Check the shell output above '
# 		'for more information.\n'
# 		'If using a hosted Colab runtime, make sure you enable GPU acceleration '
# 		'by going to the Runtime menu and selecting "Choose runtime type".')
# print('Installation successful.')    

import time
import itertools
import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List
# Graphics and plotting.
import mediapy as media
import matplotlib.pyplot as plt
# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

xml = """
<mujoco>
  <worldbody>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""

# Make model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Make renderer, render and show the pixels
renderer = mujoco.Renderer(model)
# media.show_image(renderer.render())