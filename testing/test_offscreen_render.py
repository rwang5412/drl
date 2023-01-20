import time
import numpy as np
import matplotlib.pyplot as plt
import mediapy

# Need to update env variable before import mujoco
import os
gl_option = 'egl'
os.environ['MUJOCO_GL']=gl_option
import mujoco

# Check if the env variable is correct
if "MUJOCO_GL" in os.environ:
    print(os.getenv('MUJOCO_GL'))
    print(os.getenv('PYOPENGL_PLATFORM'))

xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

size = [25, 50, 100, 150, 200, 250, 300, 350, 400, 480]
avg_list = []
for s in size:
    model = mujoco.MjModel.from_xml_path("cassie.xml")
    renderer = mujoco.Renderer(model, height=s, width=s)
    data = mujoco.MjData(model)

    print("Verify the Gl context object, ", renderer._gl_context)
    renderer.enable_depth_rendering()

    time_list = []
    frames = []
    camera_name = "egocentric"
    for i in range(1000):
        mujoco.mj_step(model, data)
        start = time.monotonic()
        renderer.update_scene(data, camera=camera_name)
        img = renderer.render().copy()
        time_list.append(time.monotonic() - start)
        img -= img.min()
        img /= 2 * img[img <= 1].mean()
        img = 255 * np.clip(img, 0, 1)
        frames.append(img.astype(np.uint8))
    mean_time = np.mean(np.array(time_list))
    print("mean time to update scene and render ", mean_time)
    avg_list.append(mean_time)
# mediapy.write_video("test.mp4", frames)
# plt.imshow(img.astype(np.uint8), cmap='gray')
# plt.colorbar(label='Distance to Camera')
# plt.show()
# print("complete")
plt.plot(size, avg_list)
plt.show()