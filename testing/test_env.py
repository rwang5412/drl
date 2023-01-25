from env import CassieEnv

def test():
  env = CassieEnv(simulator_type="mujoco", clock=None)
  env.sim.reset()
  env.sim.viewer_init()
  while env.sim.viewer.is_alive:
    if not env.sim.viewer.paused:
      for _ in range(50):
        env.sim.sim_forward()
    env.sim.viewer_render()
  print(env.sim.reset_qpos)