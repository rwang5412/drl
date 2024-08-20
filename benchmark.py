
import argparse
import itertools
import os

import torch # before wandb due to ray single core bug
import wandb
import sys

from benchmark.benchmark import Bench
from util.env_factory import env_factory
from util.file_utilities import load_args
from env.genericenv import GenericEnv


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
path = parser.parse_args().path

args = load_args(path)
print(args)

env_fn = env_factory(args.env_name, args)

# if __name__ == "__main__":

#     try:
#         evaluation_type = sys.argv[1]
#         sys.argv.remove(sys.argv[1])
#     except:
#         raise RuntimeError("Choose evaluation type from ['distributed','visual', or 'no_vis']. Or add a new one.")


# env_fn = env_factory(args['all_args'].env_name, args['env_args'])

#fix because of double logging:
# with open(os.path.join(path, "experiment.info"), 'r') as f:
#     for line in f:
#         if 'wandb_id' in line:
#             wandb_id = line.split(': ')[1].strip()

# wandb.init(
#     project=args.wandb_project_name,
#     group=args.wandb_group_name,
#     name=args.run_name,
#     resume=True,
#     id=wandb_id,
# )

# bench = Bench(path, env_fn, args=args)
# bench.run()




policy = Bench.load_policy(path, args)
bench_configs = Bench.load_configs()
# print(bench_configs)

# setup_tests(benchmark=bench_configs, env=env_fn, policy=policy)

# setup_dist_tests(bench_config=bench_configs, num_command_w= 23, num_perturb_w= 12, policy=policy, env_fn=env_fn)

# run_benchmarks(type= evaluation_type, bench_config=bench_configs, env= env_fn, policy= policy)

# command_traj_vis(env= env_fn(), policy=policy, bench_config=bench_configs)

# perturb_traj_vis(env=env_fn, policy=policy, bench_config=bench_configs)

# setup_dist_tests(benchmark=bench_configs, num_command_w=, num_perturb_w=)

# eval_perturb_trajectory(bench_config=bench_configs, env=env_fn, policy=policy)




# print(bench_configs)
# test_bench(bench_configs=bench_configs)
# sample_traj_vis(env=env_fn, policy=policy, bench_config=bench_configs)
# sample_traj_vis(env=LocomotionClockEnv, policy=policy, bench_config=bench_configs)

# perturb_test, command_test = setup_tests(bench_config=bench_configs)

# print(f"perturb test: {perturb_test}")
# print(f"command test: {command_test}")

# for test in perturb_test:
#     # print(test)  # Print the entire dictionary
#     x_force = test['schedule'][0]['x_force']
#     y_force = test['schedule'][0]['y_force']
#     z_force = test['schedule'][0]['z_force']
    # print(f"x_force: {x_force}, y_force: {y_force}, z_force: {z_force}")

# sample_traj_vis(env=env_fn, policy=policy, bench_config=bench_configs)
# eval_perturb_trajectory(env=env_fn, policy=policy, bench_config=bench_configs)





