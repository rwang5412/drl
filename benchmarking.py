
import argparse
import os

import torch # before wandb due to ray single core bug
import wandb
import pickle

from benchmark.benchmark import Bench
from util.env_factory import env_factory
from util.file_utilities import load_args


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
path = parser.parse_args().path

bench_args = load_args(path)
# print(f"args {args}")

# args = pickle.load(open(os.path.join(path, "experiment.pkl"), "rb"))
# print(f"previous_args_dict {args}")

# env_fn = env_factory(args['all_args'].env_name, args['env_args'])

args = pickle.load(open(os.path.join(path, "experiment.pkl"), "rb"))
print(args)


# env = env_factory(args['all_args'].env_name, args['env_args'])
# env_fn = env_factory(args.env_name, args)
print(f"type {type(args)}")


print(f"pkl env name {args['all_args'].env_name}")
print(f"pkl env args {args['env_args']}")

env_fn = env_factory(args['all_args'].env_name, args['env_args'])
print("env created")


# fix because of double logging:
with open(os.path.join(path, "experiment.info"), 'r') as f:
    for line in f:
        if 'wandb_id' in line:
            wandb_id = line.split(': ')[1].strip()

# print(f"wandb id {wandb_id}")

wandb.init(
    project=args['all_args'].wandb_project_name,
    group=args['all_args'].wandb_group_name,
    name=args["all_args"].run_name,
    resume=True,
    # id=wandb_id,
)

bench = Bench(path, env_fn, args=bench_args)
bench._collect_benchmarks()