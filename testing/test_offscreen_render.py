"""Test file for offscreen rendering from Mujoco.Renderer class with Cassie/Digit model.
"""

def test_offscreen_rendering():

	OFFSCREEN = 0
	gl_option = 'egl' if OFFSCREEN else 'glx'

	# Need to update env variable before import mujoco
	import os
	os.environ['MUJOCO_GL']=gl_option

	import time
	import numpy as np
	import matplotlib.pyplot as plt
	import mediapy
	from sim import MjCassieSim

	# Check if the env variable is correct
	if "MUJOCO_GL" in os.environ:
		assert os.getenv('MUJOCO_GL') == gl_option,\
			   f"GL option is {os.getenv('MUJOCO_GL')} but want to load {gl_option}."
		print("PYOPENGL_PLATFORM =",os.getenv('PYOPENGL_PLATFORM'))

	camera_name = "forward-chest-realsense-d435/depth/image-rect"
	size = [25, 50, 100, 150, 200, 250, 300, 350, 400, 480]
	# size = [400]
	sim_time_avg_list = []
	render_time_avg_list = []
	time_render_ratio = []
	for s in size:
		# Init mujoc sim and optional viewer to verify
		sim = MjCassieSim()
		sim.reset()
		sim.geom_generator._create_geom('box0', *[1, 0, 0], rise=0.5, length=0.3, width=1)
		sim.adjust_robot_pose()
		if OFFSCREEN:
			# Init renderer that reads the same model/data
			sim.init_renderer(width=s, height=s, offscreen=OFFSCREEN)
			render_state = True
		else:
			sim.viewer_init(height=1024, width=960)
			render_state = sim.viewer_render()
			# Create a second viewer that renderes a different view
			sim.init_renderer(width=400, height=400, offscreen=OFFSCREEN)

		time_raw_sim_list = []
		time_render_depth = []
		frames = []
		while render_state:
			paused = False if OFFSCREEN else sim.viewer_paused()
			if not paused:
				for _ in range(50):
					start = time.time()
					sim.sim_forward()
					time_raw_sim_list.append(time.time() - start)
			if not OFFSCREEN:
				render_state = sim.viewer_render()
			start_t = time.time()
			depth = sim.get_depth_image(camera_name)
			time_render_depth.append(time.time() - start_t)
			frames.append(depth.astype(np.uint8))
			# plt.imshow(depth)
			# plt.show()

			if sim.get_base_position()[2] < 0.5:
				mean_time_sim = np.mean(np.array(time_raw_sim_list))
				mean_time_render = np.mean(np.array(time_render_depth))
				sim_time_avg_list.append(mean_time_sim)
				render_time_avg_list.append(mean_time_render)
				time_render_ratio.append(100*mean_time_render / (mean_time_sim+mean_time_render))
				sim.renderer.close()
				break
	mediapy.write_video("test.mp4", frames, fps=50)
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
