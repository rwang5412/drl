import itertools
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import ray
import seaborn as sns
import time
import torch
import tqdm
import wandb
import json
import yaml
import sys
import json
import select

from util.nn_factory import nn_factory, load_checkpoint
from benchmark.config_generator import disturbance_rejection_configs
from util.file_utilities import load_file_by_priority
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
from ray.util import ActorPool


class Bench:
    def __init__(self, policy_path, env_fn, args):
        self.policy_path = policy_path
        self.policy = self.load_policy(policy_path, args)
        self.bench_configs: list = self.load_configs()

        ncpus = os.cpu_count() // 2 # torch.get_num_threads() is not reliable
        ray.shutdown() # in case ray is already running
        ray.init(num_cpus=torch.get_num_threads(), ignore_reinit_error=True)
        print(f"Using {ncpus} cpu cores for benchmarking...")

        print(f"ncpus {ncpus}")
        bench_workers = [BenchWorker.remote(self.policy, env_fn, i) for i in range(ncpus)]
        self.worker_pool = ray.util.ActorPool(bench_workers)
        self.env = env_fn
        self.orient_add = 0
        self.force_vector_schedule = [] # append 3dim vector to schedule a force push
        self.benchmark_force_vector = None # can be set during benching


    @staticmethod
    def load_policy(policy_path, args):
        # policy_dict = torch.load(os.path.join(policy_path, "actor.pt"))
        policy_dict = load_file_by_priority([policy_path], None, file_names_by_priority=["actor.pt", "actor_latest.pt"])
        policy, _ = nn_factory(args=args)
        load_checkpoint(model_dict=policy_dict, model=policy)
        policy.eval()
        return policy

    @staticmethod
    def load_configs():
        with open(os.path.join(os.path.dirname(__file__), 'configs.yaml'), 'r') as file:
            return yaml.safe_load(file)

    def run(self):
        results = self._collect_benchmarks()
        # self._log_benchmarks(results)

    def _collect_benchmarks(self, num_perturb_w, num_command_w):
        # Sort and prepare the tests
        perturb_tests, command_tests = sort(bench_config)
        
        # Initialize Ray
        ray.shutdown()
        ray.init(num_cpus=8)
        print("Ray initialized")
        
        # Initialize workers
        perturb_workers = [PerturbWorker.remote(self.policy, self.env, i) for i in range(num_perturb_w)]
        command_workers = [BenchWorker.remote(self.policy, self.env, i) for i in range(num_command_w)]
        print("Workers initialized")
        
        # Create actor pools
        perturb_pool = ActorPool(perturb_workers)
        command_pool = ActorPool(command_workers)
        print("pool initialized")
        
        #initialize task lists and results collection
        perturb_results = []
        command_results = []

        # Using Barts Config Generator
        for bench_config in disturbance_rejection_configs():
            # print(bench_config)
            for _ in range(self.bench_configs['n_traj']):
                perturb_pool.submit(lambda w, bench_config: w.sample_traj.remote(bench_config), bench_config)

        for bench_config in command_tests:
            command_pool.submit(lambda w, bench_config: w.sample_traj.remote(bench_config), bench_config)

        print("initial tasks done")

        #collect results as they come in
        while perturb_pool.has_next():
            output = perturb_pool.get_next_unordered()
            perturb_results.append(output)

        while command_pool.has_next():
            output = command_pool.get_next_unordered()
            command_results.append(output)

        return perturb_results, command_results
    
        # print("Collecting benchmarks...")

        # # schedule all benchmark runs
        # for bench_config in itertools.chain(self.bench_configs['benchmarks'], disturbance_rejection_configs()):
        #     for _ in range(self.bench_configs['n_traj']):
        #         self.worker_pool.submit(lambda w, bench_config: w.sample_traj.remote(bench_config), bench_config)

        # # get number of remaining jobs for progress bar
        # n_jobs_scheduled = self.worker_pool._next_task_index + len(self.worker_pool._pending_submits)
        # progress_bar = tqdm.tqdm(total=n_jobs_scheduled, smoothing=0)

        # # collect results as they come in
        # total_timesteps = 0
        # results = []
        # start = time.monotonic()
        # while self.worker_pool.has_next():
        #     timesteps, output = self.worker_pool.get_next_unordered()
        #     results.append(output)

        #     total_timesteps += timesteps
        #     progress_bar.update(1)
        #     if progress_bar.n % 1000 == 0:
        #         print(f"sampling rate: {total_timesteps / (time.monotonic() - start):.2f} timesteps/s")

        # return results

    def _log_benchmarks(self, results):
        print("Logging benchmarks...")

        df = pd.DataFrame(results)
        print(df)
        print("df columns: ", df.columns)

        self._create_timeseries_plots(df)
        self._create_disturbance_grid_plots(df)

        # TODO do error plots, error metrics. means might smooth out errors
        # TODO unified success filter
        # TODO energy efficiency metrics
        # TODO time to recovery gridplot. grey out values where successrate < 100%

    # def _create_timeseries_plots(self):
    #     wandb.init()    
    #     # Load the JSON data
    #     with open('results.json') as d:
    #         data = json.load(d)
        
    #     # Extract the list of dictionaries
    #     command_results = data['command_results']

    #     # Convert to DataFrame
    #     df = pd.DataFrame(command_results)
    #     print(f"dataframe {df}")
    #     print(type(df))

    #     names = df['name'].unique()
    #     plots = {}
    #     print(names)
    #     for name in names:
    #         # Skipping specific names as per the original function's logic
    #         if 'Ny' in name:
    #             continue
    #             # TODO temporary hack dropping disturbance rejection plots

    #         print(f"Logging {name}...")

    #         # get all rows with this name
    #         df_name = df[df['name'] == name]
    #         print(f"rows name {df_name}")

    #         # get the type of this benchmark
    #         type_ = df_name['type'].iloc[0]

    #         # get the time at which terminations happened
    #         df_terminations = df_name[df_name['termination'] == True]
    #         termination_times = (df_terminations['x_vel'].apply(len) / df_terminations['policy_rate'])
    #         print(f"termination time {termination_times}")

    #         # Select relevant columns
    #         cols = ['x_vel', 'y_vel', 'turn_rate']
    #         if all(col in df_name.columns for col in ['x_vel_cmd', 'y_vel_cmd', 'turn_rate_cmd', 'x_pos', 'y_pos', 'z_pos', 'power']):
    #             cols.extend(['x_vel_cmd', 'y_vel_cmd', 'turn_rate_cmd', 'x_pos', 'y_pos', 'z_pos', 'power'])

    #         # Plot trajectories
    #         for plot_name in ['x_vel', 'y_vel', 'turn_rate']:
    #             fig, ax = plt.subplots()
    #             for i, row in df_name.iterrows():
    #                 x = np.arange(len(row[plot_name])) / row['policy_rate']
    #                 ax.plot(x, row[plot_name], label=f"Trajectory {i+1}")

    #             ax.set_title(f"{name}: {plot_name}")
    #             ax.set_xlabel("Seconds")
    #             ax.set_ylabel(plot_name)

    #             # Add terminations histogram
    #             ax2 = ax.twinx()
    #             ax2.hist(termination_times, bins=40, alpha=0.2, color='red')
    #             ax2.yaxis.set_major_locator(MultipleLocator(1))
    #             ax2.set_ylabel("Terminations")

    #             hist_patch = mpatches.Patch(color='red', alpha=0.2, label='Terminations') # hack for legend
    #             ax.legend(handles=ax.lines + [hist_patch])

    #             plots[f"{type_.replace('_', ' ').capitalize()} Benchmark: {name}/{plot_name} img"] = wandb.Image(fig)

    #         # Log table with mean final position
    #         if 'x_pos' in cols and 'y_pos' in cols and 'z_pos' in cols:
    #             final_positions = [[row['x_pos'][-1], row['y_pos'][-1], row['z_pos'][-1]] for _, row in df_name.iterrows()]
    #             plots[f"{type_.replace('_', ' ').capitalize()} Benchmark: {name}/pos"] = \
    #                 wandb.Table(columns=["x_pos", "y_pos", "z_pos"], data=final_positions)

    #         # upload everything to wandb and local
    #         wandb.log(plots)
    #         self._save_to_disk(plots)

    def _create_timeseries_plots(self):
        wandb.init()

        # Load the JSON data
        with open('results.json') as d:
            data = json.load(d)
        
        # Extract the list of dictionaries
        command_results = data['command_results']
        
        # Convert to DataFrame
        df = pd.DataFrame(command_results)
        print(f"dataframe {df}")
        print(type(df))

        names = df['name'].unique()
        plots = {}
        print(names)
        
        for name in names:
            # Skipping specific names as per the original function's logic
            if 'Ny' in name:
                continue

            print(f"Logging {name}...")

            # Get all rows with this name
            df_name = df[df['name'] == name]
            print(f"rows name {df_name}")

            # Get the type of this benchmark
            type_ = df_name['type'].iloc[0]

            # Create a new figure and axis
            fig, ax1 = plt.subplots(figsize=(12, 6))

            # Plot x_vel, y_vel, and turn_rate
            for col in ['x_vel', 'y_vel', 'turn_rate']:
                for i, row in df_name.iterrows():
                    timesteps = np.arange(len(row[col]))
                    values = np.array(row[col])  # Convert list to NumPy array
                    if col == 'x_vel':
                        ax1.plot(timesteps, values, label='X Velocity', color='b', linewidth=2)
                    elif col == 'y_vel':
                        ax1.plot(timesteps, values, label='Y Velocity', color='g', linestyle='--', linewidth=2)
                    elif col == 'turn_rate':
                        ax1.plot(timesteps, values, label='Turn Rate', color='r', linewidth=1)

                    # Highlight overlapping lines by slightly staggering them
                    if np.allclose(values, np.array(df_name['x_vel'].iloc[i])):
                        ax1.plot(timesteps, values + 0.1, color='b', alpha=0.5)

            # Check for termination and plot a vertical line if termination is True
            termination = df_name['termination'].iloc[0] if 'termination' in df_name else False
            if termination:
                ax1.axvline(x=len(timesteps) - 1, color='k', linestyle=':', label=f'Termination at {len(timesteps) - 1}')

            ax1.set_xlabel('Timesteps')
            ax1.set_ylabel('Values')
            ax1.set_ylim(-10, 10)
            ax1.legend(loc='upper right')
            
            # Add title
            ax1.set_title(f"{name} - {type_}")

            # Save plot to WandB and local
            plots[f"{type_.replace('_', ' ').capitalize()} Benchmark: {name}"] = wandb.Image(fig)
            plt.close(fig)

        # Upload everything to WandB
        wandb.log(plots)

        # Optional: Save plots to disk
        self._save_to_disk(plots)


    def plot_trajectory(self):
        # Load data from JSON
        with open('results.json') as d:
            data = json.load(d)
            command_results = data['command_results']

        # Directory to save plots
        output_dir = os.path.join(self.policy_path, 'benchmark_plots')
        os.makedirs(output_dir, exist_ok=True)

        for entry in command_results:
            # Extract data
            name = entry['name']
            x_vel = entry['x_vel']
            y_vel = entry['y_vel']
            turn_rate = entry['turn_rate']
            termination = entry.get('termination', False)
            plot_title = entry.get('name', 'Trajectory Plot')

            # Sanitize filename
            safe_name = name.replace(" ", "_").replace("/", "_")

            # Create a new figure and axis
            fig, ax1 = plt.subplots(figsize=(12, 6))

            # Plot x velocity, y velocity, and turn rate
            ax1.plot(np.arange(len(x_vel)), x_vel, label='X Velocity', color='b', linewidth=2)
            ax1.plot(np.arange(len(y_vel)), y_vel, label='Y Velocity', color='g', linestyle='--', linewidth=2)
            ax1.plot(np.arange(len(turn_rate)), turn_rate, label='Turn Rate', color='r', linewidth=1)

            # Check for termination and plot a vertical line if termination is True
            if termination:
                ax1.axvline(x=len(x_vel)-1, color='k', linestyle=':', label=f'Termination at {len(x_vel)-1}')

            ax1.set_xlabel('Timestep')
            ax1.set_ylabel('Values')
            ax1.tick_params(axis='y')

            # Set y-axis limits
            ax1.set_ylim(-3, 8)

            # Add legends
            ax1.legend(loc='upper left')

            # Add title
            plt.title(plot_title)

            # Save the plot as a PNG file
            plot_file_path = os.path.join(output_dir, f'{safe_name}.png')
            plt.savefig(plot_file_path)

            # Close the figure to free up memory
            plt.close(fig)
            # print(f"Saved and closed plot: {plot_file_path}")


    def _create_disturbance_grid_plots(self):
        wandb.init()
        # Load the JSON data
        with open('results10traj.json') as d:
            data = json.load(d)
        
        # Extract the list of dictionaries
        perturb_results = data['perturb_results']

        # Convert to DataFrame
        df = pd.DataFrame(perturb_results)
        print(df)
        print(type(df))

        print("Disturbance grid plots...")
        
        # Filter for 'disturbance_rejection' type
        disturbance_df = df[df["type"] == "disturbance_rejection"].copy()

        # Convert to int for plotting
        disturbance_df['Nx'] = disturbance_df['Nx'].astype(int)
        disturbance_df['Ny'] = disturbance_df['Ny'].astype(int)
        disturbance_df['T'] = disturbance_df['T'].astype(int)
        n_traj = len(disturbance_df[disturbance_df['name'] == disturbance_df['name'].unique()[0]])

        unique_plots = list(set([name.split('Wz')[0]+'Wz' for name in disturbance_df['name'].unique()]))

        plots = {}
        for unique_plot in unique_plots:
            print(f"Logging {unique_plot}...")
            filtered_df = disturbance_df[disturbance_df['name'].str.contains(unique_plot)]
            for force_axis in ['Nx', 'Ny']:
                pivot_table_terminations = pd.pivot_table(filtered_df, values='termination', index=force_axis, columns='T', aggfunc="sum")
                pivot_table_terminations = pivot_table_terminations.drop(0.0)  # drop 0.0 force axis values

                pivot_table_succes_rate = 1 - (pivot_table_terminations / n_traj)
                print(pivot_table_succes_rate)

                # Create a figure and axis
                fig, ax = plt.subplots(figsize=(12, 10))

                # Creating the heatmap on the specified axes
                sns.heatmap(pivot_table_succes_rate, annot=True, fmt=".2f", cmap='RdYlGn', cbar_kws={'label': 'Success Rate'}, ax=ax)

                # Setting titles and labels
                Vx, Vy, Wz = filtered_df['Vx'].iloc[0], filtered_df['Vy'].iloc[0], filtered_df['Wz'].iloc[0]
                name = filtered_df['name'].iloc[0]
                ax.set_title(f"Disturbance Rejection Success Rates: Vx: {Vx}m/s, Vy: {Vy}m/s, Wz: {Wz}rad/s, n={n_traj} name: {name}")
                ax.set_xlabel('Impulse Length [ms]')
                ax.set_ylabel(f'Force in {force_axis[1]} direction [N]')

                plots[f'Disturbance Rejection Vx:{Vx}, Vy:{Vy}, Wz:{Wz}/{force_axis}'] = wandb.Image(fig)

        # Upload to wandb and save locally, as wandb is unreliable
        wandb.log(plots)
        self._save_to_disk(plots)



    def _save_to_disk(self, plots):
        path = os.path.join(self.policy_path, 'benchmark_plots')
        if not os.path.exists(path):
            os.makedirs(path)
        for name, plot in plots.items():
            if isinstance(plot, wandb.Image):
                plot.image.save(os.path.join(path, name.replace('/', '_')+'.png'))


    def _save_command_to_disk(self, plots):
        path = os.path.join(self.policy_path, 'benchmark_plots')
        if not os.path.exists(path):
            os.makedirs(path)
        
        for name, plot in plots.items():
            # Save each plot
            plot_file = os.path.join(path, name.replace('/', '_') + '.png')
            plot.savefig(plot_file)
            plt.close(plot)  # Close the plot to free memory



@ray.remote
class BenchWorker:
    def __init__(self, policy, env_fn, worker_id: int):
        torch.set_num_threads(1)
        self.policy = deepcopy(policy)
        self.env = env_fn()
        self.worker_id = worker_id

    def sample_traj(self, bench_config: dict):
        results = DictOfLists()
    
        with torch.no_grad():
            state = self.env.reset()
            schedule_timesteps = {round(k * self.env.default_policy_rate): v for k, v in bench_config['schedule'].items()}
            episode_length = 0
            episode_reward = []
            done = False
            state = self.env.reset()
            x_velocity = 0
            y_velocity = 0
            turn_rate = 0

            while not done:
                #step env
                state = torch.Tensor(state).float()
                action = self.policy(state).numpy()
                state, reward, done, env_infos = self.env.step(action)
                self.env.viewer_update_cop_marker()
                episode_reward.append(reward)

                infos = {
                    "x_vel": x_velocity,
                    "y_vel": y_velocity,
                    "turn_rate": turn_rate,
                }
                results.append(infos)

                if episode_length in schedule_timesteps:
                    for attr, value in schedule_timesteps[episode_length].items():
                        if attr == 'x_velocity':
                            x_velocity = value
                        if attr == 'y_velocity':
                            y_velocity = value
                        if attr == 'turn_rate':
                            turn_rate = value
                
                setattr(self.env, 'x_velocity', x_velocity)
                setattr(self.env, 'y_velocity', y_velocity)
                setattr(self.env, 'turn_rate', turn_rate)

                episode_length += 1

                if done:
                    done = True
                    break

                if episode_length >= 1200:
                    done = False
                    break

            final_results = {**bench_config, **results, 'termination': done, 'policy_rate': self.env.default_policy_rate}

            return episode_length, final_results
    

@ray.remote
class PerturbWorker(object):
    def __init__(self, policy, env_fn, worker_id: int):
        torch.set_num_threads(1)
        self.env = env_fn()
        self.worker_id = worker_id
        self.policy = deepcopy(policy)

    def sample_traj(self, bench_config: dict):
        # map seconds based schedule to timesteps # NOTE rounding might mess with values that are too close together
        schedule_timesteps = {round(k* self.env.default_policy_rate): v for k, v in bench_config['schedule'].items()}
        x_force = 0
        y_force = 0
        z_force = 0
        done = False
        with torch.no_grad():
            state = self.env.reset()
            results = DictOfLists()
            for timestep in range(bench_config['n_seconds']*self.env.default_policy_rate):

                state = torch.Tensor(state).float()
                action = self.policy(state).numpy()
                state, reward, done, infos = self.env.step(action)

                if timestep in schedule_timesteps:
                    # print(f"True at {episode_length}")
                    for attr, value in schedule_timesteps[timestep].items():
                        if attr == "force_vector":
                            # print(f"force vector {value} at step {episode_length}")
                            x_force = value[0]
                            y_force = value[1]
                            z_force = value[2]
                            # print(f"x force {x_force}, y force {y_force}, z force{z_force}")
                        else:
                            if attr == 'x_velocity':
                                x_velocity = value
                            if attr == 'y_velocity':
                                y_velocity = value
                            if attr == 'turn_rate':
                                turn_rate = value
                            # print(f"attributes set {attr} value {value} at timestep {episode_length}")

                setattr(self.env, 'x_velocity', x_velocity)
                setattr(self.env, 'y_velocity', y_velocity)
                setattr(self.env, 'turn_rate', turn_rate)
                # episode_length += 1

                set_base_force(self.env, x_force, y_force, z_force)
                # print(f"force set {x_force}, {y_force}, {z_force} at step {episode_length}")

                if done:
                    done = True
                    # print(f"max force applied of {x_force}, {y_force}, {z_force}")
                    break


        return {**bench_config, **results, 'termination': done, 'policy_rate': self.env.default_policy_rate}
    

def save_results_to_file(perturb_results, command_results, filename="results.json"):

    all_results = {
        "perturb_results": [result for result in perturb_results],
        "command_results": [result[1] for result in command_results]
    }

    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=4)


def sort(benchmark):
    perturb_tests = []
    command_tests = []

    # Process the benchmark dictionary into two lists, one for perturb tests and one for command tests
    for test in benchmark['benchmarks']:
        if test["type"] == "perturb_test":
            perturb_tests.append(test)
        elif test["type"] == "command_following":
            command_tests.append(test)
        else:
            raise ValueError("Invalid test type")
        
    # print(f"perturb_tests: {perturb_tests}")
    # print(f"command tests: {command_tests}")

    return perturb_tests, command_tests


def setup_dist_tests(bench_config, num_perturb_w, num_command_w, policy, env, n_traj):
    # Sort and prepare the tests
    perturb_tests, command_tests = sort(bench_config)
    
    # Initialize Ray
    ray.shutdown()
    ray.init(num_cpus=num_command_w + num_perturb_w)
    print("Ray initialized")
    
    # Initialize workers
    perturb_workers = [PerturbWorker.remote(policy, env, i) for i in range(num_perturb_w)]
    command_workers = [BenchWorker.remote(policy, env, i) for i in range(num_command_w)]
    print("Workers initialized")
    
    # Create actor pools
    perturb_pool = ActorPool(perturb_workers)
    command_pool = ActorPool(command_workers)
    print("pool initialized")
    
    #initialize task lists and results collection
    perturb_results = []
    command_results = []
    
    # distribute initial tasks

    for bench_config in disturbance_rejection_configs():
        # print(bench_config)
        for _ in range(n_traj):
            perturb_pool.submit(lambda w, bench_config: w.sample_traj.remote(bench_config), bench_config)

    for bench_config in command_tests:
        command_pool.submit(lambda w, bench_config: w.sample_traj.remote(bench_config), bench_config)

    print("initial tasks done")

    #collect results as they come in
    while perturb_pool.has_next():
        # done_id, remain_id = ray.wait(perturb_pool, num_returns = 1)
        # print(f" done {done_id}")
        # print(remain_id)
        output = perturb_pool.get_next_unordered()
        perturb_results.append(output)

    while command_pool.has_next():
        output = command_pool.get_next_unordered()
        command_results.append(output)

    return perturb_results, command_results


    
# Import conversions:
# Env step -> time: time per step in seconds = 1 / env.default_policy_rate (for 50 Hz, time per step = 0.02)
# Number of steps over a time period: steps = time(seconds) / time per step
# So like if a perturbation lasts 0.2 seconds and the policy rate is 50 Hz, then the number of steps would be 10 (0.2 / 0.02)
# Use this to convert all the time stuff I originally had in the perturb tests to just number of steps

def command_traj_vis(env, policy, bench_config: dict, vis_type):

    _, command_tests = sort(benchmark=bench_config)
    command_results = []

    if vis_type == 'all':

        with torch.no_grad():
            state = env.reset()
            env.sim.viewer_init()

            for command in command_tests:
                print(f"command: {command}")
                schedule_timesteps = {round(k * env.default_policy_rate): v for k, v in command['schedule'].items()}
                print(f"schedule timesteps: {schedule_timesteps}")

                episode_length = 0
                episode_reward = []
                done = False
                start_time = time.time()
                render_state = env.sim.viewer_render()
                state = env.reset()
                x_velocity = 0
                y_velocity = 0
                turn_rate = 0

                while render_state:
                    if not env.sim.viewer_paused():

                        # Step env
                        state = torch.Tensor(state).float()
                        action = policy(state).numpy()
                        state, reward, done, _ = env.step(action)
                        env.viewer_update_cop_marker()
                        episode_reward.append(reward)

                        if episode_length in schedule_timesteps:
                            # print(f"yes at {episode_length}")
                            for attr, value in schedule_timesteps[episode_length].items():
                                # print(f"attr {attr} and value {value}")
                                # setattr(env, attr, value)
                                if attr == 'x_velocity':
                                    x_velocity = value
                                    # print(f"set x to {value}")
                                    # print(x_velocity)
                                if attr == 'y_velocity':
                                    y_velocity = value
                                    # print("set y")
                                if attr == 'turn_rate':
                                    turn_rate = value
                                    # print("set turn rate")
                        
                        setattr(env, 'x_velocity', x_velocity)
                        setattr(env, 'y_velocity', y_velocity)
                        setattr(env, 'turn_rate', turn_rate)

                        # print(f"x velocity: {env.x_velocity}")
                        episode_length += 1

                        if done:
                            print(f"command {command} failed at timestep {episode_length}")
                            command_results.append(f"command {command['name']} failed at timestep {episode_length}")
                            break

                    render_state = env.sim.viewer_render()
                    delaytime = max(0, env.default_policy_rate / 2000 - (time.time() - start_time))
                    time.sleep(delaytime)

                    if episode_length == 1200 or done:
                        print(f"Episode length = {episode_length}, Average reward is {np.mean(episode_reward)}.")
                        command_results.append(f"Episode length = {episode_length}, Average reward is {np.mean(episode_reward)}.")
                        state = env.reset()
                        episode_length = 0
                        done = False
                        break
                    
    elif vis_type == 'individual':

        while True:
            input = None
            if input is None:
                input = get_user_command_input(bench_config=bench_config)
                if input is None:
                    print("Quitting the test loop.")
                    break
            with torch.no_grad():
                state = env.reset()
                env.sim.viewer_init()

                schedule_timesteps = {round(k * env.default_policy_rate): v for k, v in input['schedule'].items()}
                print(f"schedule timesteps: {schedule_timesteps}")

                episode_length = 0
                episode_reward = []
                done = False
                start_time = time.time()
                render_state = env.sim.viewer_render()
                state = env.reset()
                x_velocity = 0
                y_velocity = 0
                turn_rate = 0

                while render_state:
                    if not env.sim.viewer_paused():

                        # Step env
                        state = torch.Tensor(state).float()
                        action = policy(state).numpy()
                        state, reward, done, _ = env.step(action)
                        env.viewer_update_cop_marker()
                        episode_reward.append(reward)

                        if episode_length in schedule_timesteps:
                            for attr, value in schedule_timesteps[episode_length].items():
                                # print(f"attr {attr} and value {value}")
                                if attr == 'x_velocity':
                                    x_velocity = value
                                    # print(f"set x to {value}")
                                    # print(x_velocity)
                                if attr == 'y_velocity':
                                    y_velocity = value
                                    # print("set y")
                                if attr == 'turn_rate':
                                    turn_rate = value
                                    # print("set turn rate")
                        
                        setattr(env, 'x_velocity', x_velocity)
                        setattr(env, 'y_velocity', y_velocity)
                        setattr(env, 'turn_rate', turn_rate)

                        episode_length += 1

                        if done:
                            print(f"failed at step {episode_length}")
                            break

                    render_state = env.sim.viewer_render()
                    delaytime = max(0, env.default_policy_rate / 2000 - (time.time() - start_time))
                    time.sleep(delaytime)

                    if episode_length == 1200 or done:
                        print(f"Episode length = {episode_length}, Average reward is {np.mean(episode_reward)}.")
                        state = env.reset()
                        episode_length = 0
                        done = False
                        break


def perturb_traj_vis(env, policy, vis_type):

    with torch.no_grad():

        if vis_type == "all":
            env.sim.viewer_init()

            for perturb in disturbance_rejection_configs():
                schedule_timesteps = {round(k* env.default_policy_rate): v for k, v in perturb['schedule'].items()}
                print(f"schedule timesteps {schedule_timesteps}")
                print(f"perturbations {perturb}")

                state = env.reset()
                episode_length = 0
                render_state = env.sim.viewer_render()
                done = False
                episode_reward = []
                x_force = 0
                y_force = 0
                z_force = 0
                start_time = time.time()
            
                while render_state and not done:

                    if not env.sim.viewer_paused():
                        start_time = time.time()
                        # Step env
                        state = torch.Tensor(state).float()
                        action = policy(state).numpy()
                        state, reward, done, infos = env.step(action)
                        env.viewer_update_cop_marker()
                        episode_reward.append(reward)


                        if episode_length in schedule_timesteps:
                            print(f"True at {episode_length}")
                            for attr, value in schedule_timesteps[episode_length].items():
                                if attr == "force_vector":
                                    print(f"force vector {value} at step {episode_length}")
                                    x_force = value[0]
                                    y_force = value[1]
                                    z_force = value[2]
                                    print(f"x force {x_force}, y force {y_force}, z force{z_force}")
                                else:
                                    if attr == 'x_velocity':
                                        x_velocity = value
                                    if attr == 'y_velocity':
                                        y_velocity = value
                                    if attr == 'turn_rate':
                                        turn_rate = value
                                    print(f"attributes set {attr} value {value} at timestep {episode_length}")

                        setattr(env, 'x_velocity', x_velocity)
                        setattr(env, 'y_velocity', y_velocity)
                        setattr(env, 'turn_rate', turn_rate)
                        episode_length += 1

                        set_base_force(env, x_force, y_force, z_force)
                        # print(f"force set {x_force}, {y_force}, {z_force} at step {episode_length}")

                        if done:
                            done = True
                            print(f"max force applied of {x_force}, {y_force}, {z_force}")
                            break

                    render_state = env.sim.viewer_render()
                    delaytime = max(0, env.default_policy_rate/2000 - (time.time() - start_time))
                    time.sleep(delaytime)
                    if episode_length == 300 or done:
                        print("success")
                        done = False
                        state = env.reset()
                        episode_length = 0
                        break

        elif vis_type == 'individual':
            env.sim.viewer_init()
            while True:
                input = None
                if input is None:
                    input = get_user_perturb_input()
                    if input is None:
                        print("Quitting the test loop.")
                        break
                        
                state = env.reset()
                episode_length = 0
                render_state = env.sim.viewer_render()
                done = False
                episode_reward = []
                x_force = 0
                y_force = 0
                z_force = 0
                start_time = time.time()

                while render_state and not done:
                    schedule_timesteps = {round(k * env.default_policy_rate): v for k, v in input['schedule'].items()}

                    if not env.sim.viewer_paused():
                        start_time = time.time()
                        # Step env
                        state = torch.Tensor(state).float()
                        action = policy(state).numpy()
                        state, reward, done, infos = env.step(action)
                        env.viewer_update_cop_marker()
                        episode_reward.append(reward)

                        if episode_length in schedule_timesteps:
                            # print(f"True at {episode_length}")
                            for attr, value in schedule_timesteps[episode_length].items():
                                if attr == "force_vector":
                                    x_force = value[0]
                                    y_force = value[1]
                                    z_force = value[2]
                                else:
                                    if attr == 'x_velocity':
                                        x_velocity = value
                                    if attr == 'y_velocity':
                                        y_velocity = value
                                    if attr == 'turn_rate':
                                        turn_rate = value

                        setattr(env, 'x_velocity', x_velocity)
                        setattr(env, 'y_velocity', y_velocity)
                        setattr(env, 'turn_rate', turn_rate)
                        episode_length += 1

                        set_base_force(env, x_force, y_force, z_force)
                        print(f"force set {x_force}, {y_force}, {z_force} at step {episode_length}")

                        if done:
                            break

                    render_state = env.sim.viewer_render()
                    delaytime = max(0, env.default_policy_rate / 2000 - (time.time() - start_time))
                    time.sleep(delaytime)
                    if episode_length == 300 or done:
                        print("Test completed successfully.")
                        break
        else:
            raise ValueError("Invalid visualization type")


def get_user_perturb_input():
    print("Please enter the test name (or 'q' to quit):")
    while True:
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            input_name = sys.stdin.readline().strip()
            if input_name == 'q':
                return None
            for perturb in disturbance_rejection_configs():
                if input_name == perturb["name"]:
                    return perturb
            print("Invalid input. Please enter a valid test name (or 'q' to quit):")

def get_user_command_input(bench_config):
    _, command_tests = sort(benchmark=bench_config)
    print("Please enter the test name (or 'q' to quit):")
    while True:
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            input_name = sys.stdin.readline().strip()
            if input_name == 'q':
                return None
            for command in command_tests:
                if input_name == command["name"]:
                    return command
            print("Invalid input. Please enter a valid test name (or 'q' to quit):")

                                

def set_base_force(env, x_force, y_force, z_force):
    # """Set base force in simulation. Pops single force vector from force_vector_schedule."""
    # if benchmark_force_vector is not None:
    #     force_vector = benchmark_force_vector
    # elif len(force_vector_schedule) > 0:
    #     force_vector = force_vector_schedule.pop(0)
    # else:
    #     force_vector = np.array([0, 0, 0])
    force_vector = np.array([x_force, y_force, z_force])
    force_vector = R.from_euler(seq='xyz', angles=[0, 0, 0], degrees=False).apply(force_vector)
    base = env.sim.get_body_adr(env.sim.base_body_name)
    env.sim.data.xfrc_applied[base, :3] = force_vector

class DictOfLists(dict):
    def append(self, other):
        for key, value in other.items():
            if key not in self:
                self[key] = []
            self[key].append(value)
