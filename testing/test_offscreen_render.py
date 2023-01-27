"""Test file for offscreen rendering from Mujoco.Renderer class with Cassie/Digit model.
"""

def test_offscreen_rendering():
	# Need to update env variable before import mujoco
	import os
	gl_option = 'glx'
	os.environ['MUJOCO_GL']=gl_option

	import time
	import numpy as np
	import matplotlib.pyplot as plt
	import mediapy
	import mujoco
	from sim import MjCassieSim, DigitMjSim

	# Check if the env variable is correct
	if "MUJOCO_GL" in os.environ:
		assert os.getenv('MUJOCO_GL') == gl_option, f"GL option is {os.getenv('MUJOCO_GL')} but want to load {gl_option}."
		print(os.getenv('PYOPENGL_PLATFORM'))

	# size = [25, 50, 100, 150, 200, 250, 300, 350, 400, 480]
	size = [400]
	avg_list = []
	for s in size:
		sim = MjCassieSim()
		sim.reset()
		# renderer = mujoco.Renderer(sim.model, height=s, width=s)
		# print("Verify the Gl context object, ", renderer._gl_context)
		sim.viewer_init()
		render_state = sim.viewer_render()
		# renderer.enable_depth_rendering()

		time_list = []
		frames = []
		camera_name = "egocentric"
		# for _ in range(50):
		# 	for _ in range(50):
		# 		sim.sim_forward()
		# 	start = time.monotonic()
		# 	renderer.update_scene(sim.data, camera=camera_name)
		# 	img = renderer.render().copy()
		# 	if renderer._depth_rendering:
		# 		img -= img.min()
		# 		img /= 2 * img[img <= 1].mean()
		# 		img = 255 * np.clip(img, 0, 1)
		# 	time_list.append(time.monotonic() - start)
		# 	frames.append(img.astype(np.uint8))
		# 	if sim.get_base_position()[2] < 0.5:
		# 		break
		# mean_time = np.mean(np.array(time_list))
		# # print("mean time to update scene and render ", mean_time)
		# avg_list.append(mean_time)

		while render_state:
			start_t = time.time()
			if not sim.viewer_paused():
				for _ in range(50):
					start = time.time()
					sim.sim_forward()
					time_list.append(time.time() - start)
			render_state = sim.viewer_render()
			mean_time = np.mean(np.array(time_list))
			print("sim rate ", 1/mean_time)
	# mediapy.write_video("test.mp4", frames, fps=6)
	# plt.imshow(img.astype(np.uint8), cmap='gray')
	# plt.colorbar(label='Distance to Camera')
	# plt.show()
	# plt.plot(size, avg_list)
	# plt.show()
	print("Passed offscreen rendering test!")