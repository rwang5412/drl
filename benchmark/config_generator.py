import yaml


V_range = [
    [-0.5, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [0, -0.5, 0],
    [0, 0.5, 0],
    [0, 0, -0.4],
    [0, 0, 0.4],
]
Nx_range = [-500, -400, -300, -200, -100, 100, 200, 300, 400, 500]
Ny_range = [-500, -400, -300, -200, -100, 100, 200, 300, 400, 500]
N_range = [[Nx, 0, 0] for Nx in Nx_range] + [[0, Ny, 0] for Ny in Ny_range]
T_range = [100, 200, 300, 400, 500]

_get_config_string = lambda T, Vx, Vy, Wz, Nx, Ny: f"""
name: {Vx}Vx/{Vy}Vy/{Wz}Wz/{T}ms/{Nx}Nx/{Ny}Ny
type: disturbance_rejection
Vx: {Vx}
Vy: {Vy}
Wz: {Wz}
Nx: {Nx}
Ny: {Ny}
T: {T}
n_seconds: 8
schedule:
  0:
    x_velocity: {Vx}
    y_velocity: {Vy}
    turn_rate: 0.0
    force_vector: [0, 0, 0]
  2:
    force_vector: [{Nx}, {Ny}, 0]
  {2+T/1000}:
    force_vector: [0, 0, 0]
"""


# dynamic generator better than 13k line config file
def disturbance_rejection_configs() -> dict:
    for Vx, Vy, Wz in V_range:
        for Nx, Ny, _ in N_range:
            for T in T_range:
                yield yaml.safe_load(_get_config_string(T=T, Vx=Vx, Vy=Vy, Wz=Wz, Nx=Nx, Ny=Ny))


# main for sanity check
if __name__ == "__main__":
    i = 0
    for config_string in disturbance_rejection_configs():
        i += 1
        print(config_string)

    print(f"Total configs: {i}")
    print(f"Total timesteps: {i*100*8*50}")
    print(f"Estimated vlab sampling time: {i*100*8*50/6000/3600:.2f}h")



    with open("generated_configs.yaml", 'w') as f:
      for config in disturbance_rejection_configs():
          yaml.dump(config, f, indent=4)



