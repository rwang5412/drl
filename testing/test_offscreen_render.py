"""Test file for offscreen rendering from Mujoco.Renderer class with Cassie/Digit model.
"""

def test_offscreen_rendering():
	# Need to update env variable before import mujoco
	import os
	gl_option = 'egl'
	os.environ['MUJOCO_GL']=gl_option

	import time
	import numpy as np
	import matplotlib.pyplot as plt
	import mediapy
	import mujoco
	from sim import MjCassieSim

	# Check if the env variable is correct
	if "MUJOCO_GL" in os.environ:
		assert os.getenv('MUJOCO_GL') == gl_option, f"GL option is {os.getenv('MUJOCO_GL')} but want to load {gl_option}."
		print(os.getenv('PYOPENGL_PLATFORM'))

	size = [25, 50, 100, 150, 200, 250, 300, 350, 400, 480]
	# size = [400]
	sim_time_avg_list = []
	render_time_avg_list = []
	time_render_ratio = []
	for s in size:
		# Init mujoc sim and optional viewer to verify
		sim = MjCassieSim()
		sim.reset()
		camera_name = "forward-chest-realsense-d435/depth/image-rect"
		# sim.viewer_init(camera_id=camera_name, height=500, width=500)
		# render_state = sim.viewer_render()

		# Init renderer that reads the same model/data
		renderer = mujoco.Renderer(sim.model, height=s, width=s)
		print("Verify the Gl context object, ", renderer._gl_context)
		renderer.enable_depth_rendering()

		time_raw_sim_list = []
		time_render_depth = []
		frames = []
		while True:
			# if not sim.viewer_paused():
			for _ in range(50):
				start = time.time()
				sim.sim_forward()
			time_raw_sim_list.append(time.time() - start)
			# render_state = sim.viewer_render()
			# Render offscreen
			start_t = time.time()
			renderer.update_scene(sim.data, camera=camera_name)
			img = renderer.render()
			time_render_depth.append(time.time() - start_t)
			frames.append(img)

			if sim.get_base_position()[2] < 0.5:
				mean_time_sim = np.mean(np.array(time_raw_sim_list))
				mean_time_render = np.mean(np.array(time_render_depth))
				sim_time_avg_list.append(mean_time_sim)
				render_time_avg_list.append(mean_time_render)
				time_render_ratio.append(100*mean_time_render / (mean_time_sim+mean_time_render))
				break
	mediapy.write_video("test.mp4", frames, fps=30)
	fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2)
	ax1.plot(size, sim_time_avg_list)
	ax1.set_title('sim time [s] per policy step')
	ax2.plot(size, render_time_avg_list)
	ax2.set_title('render time [s] per policy step')
	ax3.plot(size, time_render_ratio)
	ax3.set_title('render ratio [%] per policy step')
	wall_time = [t * 300e6 / 3600 for t in render_time_avg_list]
	ax4.plot(size, wall_time)
	ax4.set_title('additional walk clock time [hours]\nfor 300M samples')
	ax4.set_xlabel('image size')
	fig.tight_layout()
	plt.show()
	print("Passed offscreen rendering test!")
