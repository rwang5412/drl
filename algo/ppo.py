"""Proximal Policy Optimization (clip objective)."""
import argparse
import numpy as np
import os
import ray
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from copy import deepcopy
from functools import reduce
from operator import add
from time import time, sleep
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence
from torch.nn.utils.rnn import pad_sequence
from types import SimpleNamespace

from algo.util.sampling import Buffer, AlgoSampler
from algo.util.worker import AlgoWorker
from util.mirror import mirror_tensor


@ray.remote
class PPOOptim(AlgoWorker):
    """
        Worker for doing optimization step of PPO.

        Args:
            actor: actor pytorch network
            critic: critic pytorch network
            a_lr (float): actor learning rate
            c_lr (float): critic learning rate
            eps (float): adam epsilon
            entropy_coeff (float): entropy regularizaiton coefficient
            grad_clip (float): Value to clip gradients at.
            mirror (int or float): scalar multiple of mirror loss
            clip (float): Clipping parameter for PPO surrogate loss

        Attributes:
            actor: actor pytorch network
            critic: critic pytorch network
    """
    def __init__(self,
                 actor,
                 critic,
                 a_lr=1e-4,
                 c_lr=1e-4,
                 eps=1e-6,
                 entropy_coeff=0,
                 grad_clip=0.01,
                 mirror=0,
                 clip=0.2,
                 save_path=None,
                 **kwargs):
        AlgoWorker.__init__(self, actor, critic)
        self.old_actor = deepcopy(actor)
        self.actor_optim   = optim.Adam(self.actor.parameters(), lr=a_lr, eps=eps)
        self.critic_optim  = optim.Adam(self.critic.parameters(), lr=c_lr, eps=eps)
        self.entropy_coeff = entropy_coeff
        self.grad_clip = grad_clip
        self.mirror    = mirror
        self.clip = clip
        self.save_path = save_path

    def optimize(self,
                 memory,
                 epochs=4,
                 batch_size=32,
                 kl_thresh=0.02,
                 recurrent=False,
                 state_mirror_indices=None,
                 action_mirror_indices=None,
                 verbose=False):
        """
        Does a single optimization step given buffer info

        Args:
            memory (Buffer): Buffer object of rollouts from experience collection phase of PPO
            epochs (int): optimization epochs
            batch_size (int): optimization batch size
            kl_thresh (float): threshold for max kl divergence
            recurrent (bool): Buffer samples for recurrent policy or not
            state_mirror_indices(list): environment-specific list of mirroring information
            state_mirror_indices(list): environment-specific list of mirroring information
            verbose (bool): verbose logger output
        """
        self.old_actor.load_state_dict(self.actor.state_dict())
        torch.set_num_threads(1)
        kls, a_loss, c_loss, m_loss = [], [], [], []
        done = False
        state_mirror_indices =  state_mirror_indices if self.mirror > 0 else None
        for epoch in range(epochs):
            epoch_start = time()
            for batch in memory.sample(batch_size=batch_size,
                                       recurrent=recurrent,
                                       mirror_state_idx=state_mirror_indices):

                if state_mirror_indices is not None:
                    states, mirror_states, actions, returns, advantages, mask = batch
                else:
                    mirror_states = None
                    states, actions, returns, advantages, mask = batch

                start = time()
                kl, losses = self._update_policy(states,
                                                 actions,
                                                 returns,
                                                 advantages,
                                                 mask,
                                                 mirror_states=mirror_states,
                                                 mirror_action_idx=action_mirror_indices)
                kls    += [kl]
                a_loss += [losses[0]]
                c_loss += [losses[1]]
                m_loss += [losses[2]]

                if max(kls) > kl_thresh:
                    print(f"\t\tbatch had kl of {max(kls)} (threshold {kl_thresh}), stopping " \
                          f"optimization early.")
                    done = True
                    break

            if verbose:
                print(f"\t\tepoch {epoch+1:2d} in {(time() - epoch_start):3.2f}s, " \
                      f"kl {np.mean(kls):6.5f}, actor loss {np.mean(a_loss):6.3f}, " \
                      f"critic loss {np.mean(c_loss):6.3f}")

            if done:
                break
        return np.mean(a_loss), np.mean(c_loss), np.mean(m_loss), np.mean(kls)

    def retrieve_parameters(self):
        """
        Function to return parameters for optimizer copies of actor and critic
        """
        return list(self.actor.parameters()), list(self.critic.parameters())

    def _update_policy(self,
                       states,
                       actions,
                       returns,
                       advantages,
                       mask,
                       mirror_states=None,
                       mirror_action_idx=None):
        with torch.no_grad():
            old_pdf       = self.old_actor.pdf(states)
            old_log_probs = old_pdf.log_prob(actions).sum(-1, keepdim=True)

        # get new action distribution and log probabilities
        pdf       = self.actor.pdf(states)
        log_probs = pdf.log_prob(actions).sum(-1, keepdim=True)

        ratio      = ((log_probs - old_log_probs) * mask).exp()
        cpi_loss   = ratio * advantages * mask
        clip_loss  = ratio.clamp(1.0 - self.clip, 1 + self.clip) * advantages * mask
        actor_loss = -torch.min(cpi_loss, clip_loss).mean()

        critic_loss = 0.5 * ((returns - self.critic(states)) * mask).pow(2).mean()

        entropy_penalty = -(self.entropy_coeff * pdf.entropy() * mask).mean()

        if self.mirror > 0 and mirror_states is not None and mirror_action_idx is not None:
            mirror_time = time()
            with torch.no_grad():
                mirrored_actions = mirror_tensor(self.actor(mirror_states), mirror_action_idx)

            unmirrored_actions = pdf.mean
            mirror_loss = self.mirror * 4 * (unmirrored_actions - mirrored_actions).pow(2).mean()
        else:
            mirror_loss = torch.zeros(1)

        if not (torch.isfinite(states).all() and torch.isfinite(actions).all() \
                and torch.isfinite(returns).all() and torch.isfinite(advantages).all() \
                and torch.isfinite(log_probs).all() and torch.isfinite(old_log_probs).all() \
                and torch.isfinite(actor_loss).all() and torch.isfinite(critic_loss).all() \
                and torch.isfinite(mirror_loss).all()):
            torch.save({"states": states,
                        "actions": actions,
                        "returns": returns,
                        "advantages": advantages,
                        "log_probs": log_probs,
                        "old_log_probs": old_log_probs,
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss,
                        "mirror_loss": mirror_loss,
                        "pdf": pdf,
                        "old pdf": old_pdf}, os.path.join(self.save_path, "training_error.pt"))
            raise RuntimeError(f"Optimization experiences non-finite values, please check locally"
                               f" saved file at training_error.pt for further diagonose.")

        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()

        (actor_loss + entropy_penalty + mirror_loss).backward()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip)
        self.actor_optim.step()
        self.critic_optim.step()

        with torch.no_grad():
          kl = kl_divergence(pdf, old_pdf).mean().numpy()

          return kl, ((actor_loss + entropy_penalty).item(), critic_loss.item(), mirror_loss.item())


class PPO(AlgoWorker):
    """
    Worker for sampling experience for PPO

    Args:
        actor: actor pytorch network
        critic: critic pytorch network
        env_fn: environment constructor function
        args: argparse namespace


    Attributes:
        actor: actor pytorch network
        critic: critic pytorch network
        recurrent: recurrent policies or not
        env_fn: environment constructor function
        discount: discount factor
        entropy_coeff: entropy regularization coeff

        grad_clip: value to clip gradients at
        mirror: scalar multiple of mirror loss. No mirror loss if this equals 0
        env: instance of environment
        state_mirror_indices (func): environment-specific function for mirroring state for mirror loss
        action_mirror_indices (func): environment-specific function for mirroring action for mirror loss
        workers (list): list of ray worker IDs for sampling experience
        optim: ray woker ID for optimizing

    """
    def __init__(self, actor, critic, env_fn, args):

        self.actor = actor
        self.critic = critic
        AlgoWorker.__init__(self, actor, critic)

        if actor.is_recurrent or critic.is_recurrent:
            self.recurrent = True
        else:
            self.recurrent = False

        self.env_fn        = env_fn
        self.discount      = args.discount
        self.entropy_coeff = args.entropy_coeff
        self.grad_clip     = args.grad_clip
        self.mirror        = args.mirror
        self.env           = env_fn()

        if not ray.is_initialized():
            if args.redis is not None:
                ray.init(redis_address=args.redis)
            else:
                ray.init(num_cpus=args.workers)

        self.state_mirror_indices = None
        self.action_mirror_indices = None
        if hasattr(self.env, 'get_state_mirror_indices'):
            self.state_mirror_indices = self.env.get_state_mirror_indices()
        if hasattr(self.env, 'get_action_mirror_indices'):
            self.action_mirror_indices = self.env.get_action_mirror_indices()

        self.workers = [AlgoSampler.remote(actor, critic, env_fn, args.discount, i) for i in \
                        range(args.workers)]
        self.optim   = PPOOptim.remote(actor, critic, **vars(args))

    def do_iteration(self,
                     num_steps,
                     max_traj_len,
                     epochs,
                     kl_thresh=0.02,
                     verbose=True,
                     batch_size=64,
                     mirror=False):
        """
        Function to do a single iteration of PPO

        Args:
            max_traj_len (int): maximum trajectory length of an episode
            num_steps (int): number of steps to collect experience for
            epochs (int): optimzation epochs
            batch_size (int): optimzation batch size
            mirror (bool): Mirror loss enabled or not
            kl_thresh (float): threshold for max kl divergence
            verbose (bool): verbose logging output
        """
        copy_start = time()
        times = {}
        actor_param_id  = ray.put(list(self.actor.parameters()))
        critic_param_id = ray.put(list(self.critic.parameters()))
        norm_id = ray.put([self.actor.welford_state_mean, self.actor.welford_state_mean_diff, \
                           self.actor.welford_state_n])

        steps = max(num_steps // len(self.workers), max_traj_len)

        for w in self.workers:
            w.sync_policy.remote(actor_param_id, critic_param_id, input_norm=norm_id)

        if verbose:
            print("\t{:5.4f}s to copy policy params to workers.".format(time() - copy_start))
        torch.set_num_threads(1)

        eval_buffers, _, _ = zip(*ray.get([w.sample_traj.remote(max_traj_len=max_traj_len, \
                                           do_eval=True) for w in self.workers]))
        eval_memory = reduce(add, eval_buffers)
        eval_reward = np.mean(eval_memory.rewards)
        avg_ep_len = np.mean(eval_memory.ep_lens)

        # Sampling for optimization
        sampling_start = time()
        sampled_steps = 0
        avg_efficiency = 0
        num_traj = 0
        memory = None
        sample_jobs = [w.sample_traj.remote(max_traj_len) for w in self.workers]
        while sampled_steps < num_steps:
            done_id, remain_id = ray.wait(sample_jobs, num_returns = 1)
            buf, efficiency, work_id = ray.get(done_id)[0]
            if memory is None:
                memory = buf
            else:
                memory += buf
            num_traj += 1
            sampled_steps += len(buf)
            avg_efficiency += (efficiency - avg_efficiency) / num_traj
            sample_jobs[work_id] = self.workers[work_id].sample_traj.remote(max_traj_len)

        map(ray.cancel, sample_jobs) # Cancel leftover unneeded jobs

        total_steps = len(memory)
        avg_batch_reward = np.mean(memory.ep_returns)
        sampling_elapsed = time() - sampling_start
        sample_rate = (total_steps / 1000) / sampling_elapsed
        ideal_efficiency = avg_efficiency * len(self.workers)
        times["Sample Time"] = sampling_elapsed
        times["Sample Rate"] = sample_rate
        times["Ideal Sample Rate"] = ideal_efficiency / 1000
        times["Overhead Loss"] = sampling_elapsed - total_steps / ideal_efficiency
        if verbose:
            print(f"\t{sampling_elapsed:3.2f}s to collect {total_steps:6n} timesteps | " \
                  f"{sample_rate:3.2}k/s.")
            print(f"\tIdealized efficiency {times['Ideal Sample Rate']:3.2f}k/s \t | Time lost to " \
                  f"overhead {times['Overhead Loss']:.2f}s")

        if self.mirror > 0 and self.state_mirror_indices is not None and \
           self.action_mirror_indices is not None:
            state_mirror_indices = self.state_mirror_indices
            action_mirror_indices = self.action_mirror_indices
        else:
            state_mirror_indices = None
            action_mirror_indices = None

        optim_start = time()
        self.optim.sync_policy.remote(actor_param_id, critic_param_id, input_norm=norm_id)
        losses = ray.get(self.optim.optimize.remote(memory,
                                                    epochs=epochs,
                                                    batch_size=batch_size,
                                                    kl_thresh=kl_thresh,
                                                    recurrent=self.recurrent,
                                                    state_mirror_indices=state_mirror_indices,
                                                    action_mirror_indices=action_mirror_indices,
                                                    verbose=verbose))
        actor_params, critic_params = ray.get(self.optim.retrieve_parameters.remote())
        a_loss, c_loss, m_loss, kls = losses
        self.sync_policy(actor_params, critic_params)
        times["Optimize Time"] = time() - optim_start
        if verbose:
            print(f"\t{times['Optimize Time']:3.2f}s to update policy.")
        return eval_reward, np.mean(kls), np.mean(a_loss), np.mean(c_loss), np.mean(m_loss), \
               len(memory), times, total_steps, avg_ep_len, avg_batch_reward

def add_algo_args(parser):
    default_values = {
        "prenormalize-steps" : (100, "Number of steps to use in prenormlization"),
        "prenorm"            : (False, "Whether to do prenormalization or not"),
        "update-norm"        : (False, "Update input normalization during training."),
        "num_steps"          : (5000, "Number of steps to sample each iteration"),
        "discount"           : (0.99, "Discount factor when calculating returns"),
        "a-lr"               : (1e-4, "Actor policy learning rate"),
        "c-lr"               : (1e-4, "Critic learning rate"),
        "eps"                : (1e-6, "Adam optimizer eps value"),
        "kl"                 : (0.02, "KL divergence threshold"),
        "entropy-coeff"      : (0.0, "Coefficient of entropy loss in optimization"),
        "clip"               : (0.2, "Log prob clamping value (1 +- clip)"),
        "grad-clip"          : (0.05, "Gradient clip value (maximum allowed gradient norm)"),
        "batch-size"         : (64, "Minibatch size to use during optimization"),
        "epochs"             : (3, "Number of epochs to optimize for each iteration"),
        "mirror"             : (0, "Mirror loss coefficient"),
        "workers"            : (4, "Number of parallel workers to use for sampling"),
        "redis"              : (None, "Ray redis address"),
        "previous"           : ("", "Previous model to bootstrap from"),
    }
    if isinstance(parser, argparse.ArgumentParser):
        ppo_group = parser.add_argument_group("PPO arguments")
        for arg, (default, help_str) in default_values.items():
            if isinstance(default, bool):   # Arg is bool, need action 'store_true' or 'store_false'
                ppo_group.add_argument("--" + arg, default = default, action = "store_" + \
                                    str(not default).lower(), help = help_str)
            else:
                ppo_group.add_argument("--" + arg, default = default, type = type(default),
                                      help = help_str)
    elif isinstance(parser, SimpleNamespace) or isinstance(parser, argparse.Namespace()):
        for arg, (default, help_str) in default_values.items():
            arg = arg.replace("-", "_")
            if not hasattr(parser, arg):
                setattr(parser, arg, default)

    return parser


def run_experiment(parser, env_name):
    """
    Function to run a PPO experiment.

    Args:
        parser: argparse object
    """
    from algo.util.normalization import train_normalizer
    from algo.util.log import create_logger
    from util.env_factory import env_factory, add_env_parser
    from util.nn_factory import nn_factory, load_checkpoint, save_checkpoint, add_nn_parser
    from util.colors import FAIL, ENDC, WARNING

    import pickle
    import locale
    locale.setlocale(locale.LC_ALL, '')

    # Add env and NN parser arguments, then can finally parse args
    if isinstance(parser, argparse.ArgumentParser):
        add_env_parser(env_name, parser)
        add_nn_parser(parser)
        args = parser.parse_args()
        for arg_group in parser._action_groups:
            if arg_group.title == "PPO arguments":
                ppo_dict = {a.dest: getattr(args, a.dest, None) for a in arg_group._group_actions}
                ppo_args = argparse.Namespace(**ppo_dict)
            elif arg_group.title == "Env arguments":
                env_dict = {a.dest: getattr(args, a.dest, None) for a in arg_group._group_actions}
                env_args = argparse.Namespace(**env_dict)
            elif arg_group.title == "NN arguments":
                nn_dict = {a.dest: getattr(args, a.dest, None) for a in arg_group._group_actions}
                nn_args = argparse.Namespace(**nn_dict)
    elif isinstance(parser, SimpleNamespace) or isinstance(parser, argparse.Namespace):
        env_args = SimpleNamespace()
        nn_args = SimpleNamespace()
        ppo_args = parser
        add_env_parser(env_name, env_args)
        add_nn_parser(nn_args)
        args = parser
    else:
        raise RuntimeError(f"{FAIL}ppo.py run_experiment got invalid object type for arguments. " \
                           f"Input object should be either an ArgumentParser or a " \
                           f"SimpleNamespace.{ENDC}")

    # wrapper function for creating parallelized envs
    env_fn = env_factory(env_name, env_args)
    args.env_name = env_name # add back in since deleted earlier

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Process args for NN. Append more in here and add_nn_parser() if needed
    nn_args.obs_dim = env_fn().observation_size
    nn_args.action_dim = env_fn().action_size
    nn_args.std = torch.ones(nn_args.action_dim)*nn_args.std
    nn_args.layers = [int(x) for x in nn_args.layers.split(',')]
    nn_args.nonlinearity = getattr(torch, nn_args.nonlinearity)

    if args.std_array:
        assert len(args.std_array) == nn_args.action_dim,\
               f"{FAIL}Std noise array size mismatch with action size.{ENDC}"
        nn_args.std = torch.tensor(args.std_array)

    policy, critic = nn_factory(args=nn_args)
    # Load model attributes if args.previous exists
    if hasattr(args, "previous") and args.previous != "":
        actor_dict = torch.load(os.path.join(args.previous, "actor.pt"))
        critic_dict = torch.load(os.path.join(args.previous, "critic.pt"))
        load_checkpoint(model_dict=actor_dict, model=policy)
        load_checkpoint(model_dict=critic_dict, model=critic)
        prev_args_dict = pickle.load(open(os.path.join(args.previous, "experiment.pkl"), "rb"))
        # Compare if any arg has been changed (add/remove/update)
        for a in vars(nn_args):
            if a in prev_args_dict['nn_args']:    
                try:
                    if getattr(nn_args, a) != getattr(prev_args_dict['nn_args'], a):
                        print(f"{WARNING}Argument {a} is set to a new value {getattr(nn_args, a)}, "
                              f"old one is {getattr(prev_args_dict['nn_args'], a)}.{ENDC}")
                except:
                    if getattr(nn_args, a).any() != getattr(prev_args_dict['nn_args'], a).any():
                        print(f"{WARNING}Argument {a} is set to a new value {getattr(nn_args, a)}, "
                              f"old one is {getattr(prev_args_dict['nn_args'], a)}.{ENDC}")
            else:
                print(f"{WARNING}Added a new argument: {a}.{ENDC}")
        exit()
        # Update with new args
        for arg in nn_args:
            if arg not in ["obs_dim", "action_dim", "layers", "bounded", "nonlinearity"]:
                setattr(policy, arg, getattr(nn_args, arg))
                print(f"Set {arg} with {getattr(nn_args, arg)}. Previous as ")
    # Prenormalization only on new training
    if args.prenorm and args.previous == "":
        print("Collecting normalization statistics with {} states...".format(args.prenormalize_steps))
        train_normalizer(env_fn, policy, args.prenormalize_steps, max_traj_len=args.traj_len, noise=1)
        critic.copy_normalizer_stats(policy)

    policy.train(True)
    critic.train(True)
    # create actor/critic dict tp include model_state_dict and other class attributes
    actor_dict = {'model_class_name': policy._get_name()}
    critic_dict = {'model_class_name': critic._get_name()}

    # create a tensorboard logging object
    # before create logger files, double check that all args are updated in case any other of
    # ppo_args, env_args, nn_args changed
    for arg in ppo_args.__dict__:
        setattr(args, arg, getattr(ppo_args, arg))
    for arg in env_args.__dict__:
        setattr(args, arg, getattr(env_args, arg))
    for arg in nn_args.__dict__:
        setattr(args, arg, getattr(nn_args, arg))
    logger = create_logger(args, ppo_args, env_args, nn_args)
    args.save_actor_path = os.path.join(logger.dir, 'actor.pt')
    args.save_critic_path = os.path.join(logger.dir, 'critic.pt')
    args.save_path = logger.dir

    # Algo init
    algo = PPO(policy, critic, env_fn, ppo_args)

    print("Proximal Policy Optimization:")
    for key, val in args.__dict__.items():
        print(f"\t{key} = {val}")

    itr = 0
    timesteps = 0
    best_reward = None
    past500_reward = -1
    while timesteps < args.timesteps:
        eval_reward, kl, a_loss, c_loss, m_loss, steps, times, total_steps, avg_ep_len, avg_batch_reward = \
            algo.do_iteration(args.num_steps, args.traj_len, args.epochs, batch_size=args.batch_size, kl_thresh=args.kl, mirror=args.mirror)

        timesteps += steps
        print(f"iter {itr:4d} | return: {eval_reward:5.2f} | KL {kl:5.4f} | Actor loss {a_loss:5.4f}" \
              f" | Critic loss {c_loss:5.4f} | ", end='')
        if m_loss != 0:
            print(f"mirror {m_loss:6.5f} | ", end='')

        print(f"timesteps {timesteps:n}")

        # Savhing checkpoints for best reward
        if best_reward is None or eval_reward > best_reward:
            print(f"\t(best policy so far! saving checkpoint to {args.save_actor_path})")
            best_reward = eval_reward
            save_checkpoint(algo.actor, actor_dict, args.save_actor_path)
            save_checkpoint(algo.critic, critic_dict, args.save_critic_path)

        if itr > 2:
            exit()
        # Intermitent saving
        if itr % 500 == 0:
            past500_reward = -1
        if eval_reward > past500_reward:
            past500_reward = eval_reward
            save_checkpoint(algo.actor, actor_dict, args.save_actor_path[:-3] + "_past500.pt")
            save_checkpoint(algo.critic, critic_dict, args.save_critic_path[:-3] + "_past500.pt")

        if logger is not None:
            logger.add_scalar("Test/Return", eval_reward, itr)
            logger.add_scalar("Train/Return", avg_batch_reward, itr)
            logger.add_scalar("Train/Mean Eplen", avg_ep_len, itr)
            logger.add_scalar("Train/Mean KL Div", kl, itr)

            logger.add_scalar("Misc/Critic Loss", c_loss, itr)
            logger.add_scalar("Misc/Actor Loss", a_loss, itr)
            logger.add_scalar("Misc/Mirror Loss", m_loss, itr)
            logger.add_scalar("Misc/Timesteps", total_steps, itr)

            for time, val in times.items():
                logger.add_scalar("Misc/" + time, val, itr)

        itr += 1
    print(f"Finished ({timesteps} of {args.timesteps}).")

    if args.wandb:
        wandb.finish()
