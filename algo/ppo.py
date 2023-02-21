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
from time import time, sleep
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence
from torch.nn.utils.rnn import pad_sequence
from types import SimpleNamespace

from util.mirror import mirror_tensor

class Buffer:
    """
    Generic Buffer class to hold samples for PPO.
    Note: that it is assumed that trajectories are stored
    consecutively, next to each other. The list traj_idx stores the indices where individual
    trajectories are started.

    Args:
        discount (float): Discount factor

    Attributes:
        states (list): List of stored sampled observation states
        actions (list): List of stored sampled actions
        rewards (list): List of stored sampled rewards
        values (list): List of stored sampled values
        returns (list): List of stored computed returns
        advantages (list): List of stored computed advantages
        ep_returns (list): List of trajectories returns (summed rewards over whole trajectory)
        ep_lens (list): List of trajectory lengths
        size (int): Number of currently stored states
        traj_idx (list): List of indices where individual trajectories start
        buffer_read (bool): Whether or not the buffer is ready to be used for optimization.
    """
    def __init__(self, discount=0.99):
        self.discount = discount
        self.clear()

    def __len__(self):
        return len(self.states)

    def clear(self):
        """
        Clear out/reset all buffer values. Should always be called before starting new sampling iteration
        """
        self.states     = []
        self.actions    = []
        self.rewards    = []
        self.values     = []
        self.returns    = []
        self.advantages = []

        self.ep_returns = []
        self.ep_lens = []

        self.size = 0

        self.traj_idx = [0]
        self.buffer_ready = False

    def push(self, state, action, reward, value, done=False):
        """
        Store new PPO state (state, action, reward, value, termination)

        Args:
            state (numpy vector):  observation
            action (numpy vector): policy action
            reward (numpy vector): reward
            value (numpy vector): value function value
            return (numpy vector): return
            done (bool): last mdp tuple in rollout
        """
        self.states  += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values  += [value]

        self.size += 1

    def end_trajectory(self, terminal_value=0):
        """
        Finish a stored trajectory, i.e. calculate return for each step by adding a termination
        value to the last state and backing up return based on discount factor.

        Args:
            terminal_value (float): Estimated value at the final state in the trajectory. Used to
                                    back up and calculate returns for the whole trajectory
        """
        self.traj_idx += [self.size]
        rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]

        returns = []

        R = terminal_value
        for reward in reversed(rewards):
            R = self.discount * R + reward
            returns.insert(0, R)

        self.returns += returns

        self.ep_returns += [np.sum(rewards)]
        self.ep_lens    += [len(rewards)]

    def _finish_buffer(self, state_mirror_idx):
        """
        Get a buffer ready for optimization by turning each list into torch Tensor. Also calculate
        mirror states and normalized advantages. Must be called before sampling from the buffer for
        optimization. While make "buffer_ready" variable true.

        Args:
            mirror (function pointer): Pointer to the state mirroring function that while mirror
                                       observation states
        """
        with torch.no_grad():
            self.states  = torch.Tensor(np.array(self.states))
            self.actions = torch.Tensor(np.array(self.actions))
            self.rewards = torch.Tensor(np.array(self.rewards))
            self.returns = torch.Tensor(np.array(self.returns))
            self.values  = torch.Tensor(np.array(self.values))

            # Mirror states in needed
            if state_mirror_idx is not None:
                self.mirror_states = mirror_tensor(self.states, state_mirror_idx)

            # Calculate and normalize advantages
            a = self.returns - self.values
            a = (a - a.mean()) / (a.std() + 1e-4)
            self.advantages = a
            self.buffer_ready = True

    def sample(self, batch_size=64, recurrent=False, mirror_state_idx=None):
        """
        Returns a randomly sampled batch from the buffer to be used for optimization. If "recurrent"
        is true, will return a random batch of trajectories to be used for backprop through time.
        Otherwise will return randomly selected states from the buffer.

        Args:
            batch_size (int): Size of the batch. If recurrent is True then the number of
                              trajectories to return. Otherwise is the number of states to return.
            recurrent (bool): Whether to return a recurrent batch (trajectories) or not
            mirror (function pointer): Pointer to the state mirroring function. If is None, the no
                                       mirroring will be done.
        """
        if not self.buffer_ready:
            self._finish_buffer(mirror_state_idx)

        if recurrent:
            random_indices = SubsetRandomSampler(range(len(self.traj_idx)-1))
            sampler = BatchSampler(random_indices, batch_size, drop_last=False)

            for traj_indices in sampler:
                states     = [self.states[self.traj_idx[i]:self.traj_idx[i+1]]     for i in traj_indices]
                actions    = [self.actions[self.traj_idx[i]:self.traj_idx[i+1]]    for i in traj_indices]
                returns    = [self.returns[self.traj_idx[i]:self.traj_idx[i+1]]    for i in traj_indices]
                advantages = [self.advantages[self.traj_idx[i]:self.traj_idx[i+1]] for i in traj_indices]
                traj_mask  = [torch.ones_like(r) for r in returns]

                states     = pad_sequence(states,     batch_first=False)
                actions    = pad_sequence(actions,    batch_first=False)
                returns    = pad_sequence(returns,    batch_first=False)
                advantages = pad_sequence(advantages, batch_first=False)
                traj_mask  = pad_sequence(traj_mask,  batch_first=False)

                if mirror_state_idx is None:
                    yield states, actions, returns, advantages, traj_mask
                else:
                    mirror_states = [self.mirror_states[self.traj_idx[i]:self.traj_idx[i+1]] for i in traj_indices]
                    mirror_states = pad_sequence(mirror_states, batch_first=False)
                    yield states, mirror_states, actions, returns, advantages, traj_mask

        else:
            random_indices = SubsetRandomSampler(range(self.size))
            sampler = BatchSampler(random_indices, batch_size, drop_last=True)

            for i, idxs in enumerate(sampler):
                states     = self.states[idxs]
                actions    = self.actions[idxs]
                returns    = self.returns[idxs]
                advantages = self.advantages[idxs]

                if mirror_state_idx is None:
                    yield states, actions, returns, advantages, 1
                else:
                    mirror_states = self.mirror_states[idxs]
                    yield states, mirror_states, actions, returns, advantages, 1

def merge_buffers(buffers):
    """
    Function to merge a list of buffers into a single Buffer object. Used for merging buffers
    received from multiple remote workers into a simple Buffer object to sample from.

    Args:
        buffers (list): List of Buffer objects to merge

    Returns:
        A single Buffer object
    """
    memory = Buffer()

    for b in buffers:
        offset = len(memory)

        memory.states  += b.states
        memory.actions += b.actions
        memory.rewards += b.rewards
        memory.values  += b.values
        memory.returns += b.returns

        memory.ep_returns += b.ep_returns
        memory.ep_lens    += b.ep_lens

        memory.traj_idx += [offset + i for i in b.traj_idx[1:]]
        memory.size     += b.size

    return memory

class PPO_Worker:
    """
        Generic template for a worker (sampler or optimizer) for PPO

        Args:
            actor: actor pytorch network
            critic: critic pytorch network

        Attributes:
            actor: actor pytorch network
            critic: critic pytorch network
    """
    def __init__(self, actor, critic):
        self.actor = deepcopy(actor)
        self.critic = deepcopy(critic)

    def sync_policy(self, new_actor_params, new_critic_params, input_norm=None):
        """
        Function to sync the actor and critic parameters with new parameters.

        Args:
            new_actor_params (torch dictionary): New actor parameters to copy over
            new_critic_params (torch dictionary): New critic parameters to copy over
            input_norm (int): Running counter of states for normalization
        """
        for p, new_p in zip(self.actor.parameters(), new_actor_params):
            p.data.copy_(new_p)

        for p, new_p in zip(self.critic.parameters(), new_critic_params):
            p.data.copy_(new_p)

        if input_norm is not None:
            self.actor.welford_state_mean, self.actor.welford_state_mean_diff, self.actor.welford_state_n = input_norm
            self.critic.copy_normalizer_stats(self.actor)

@ray.remote
class PPO_Optim(PPO_Worker):
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
        PPO_Worker.__init__(self, actor, critic)
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
            for batch in memory.sample(batch_size=batch_size, recurrent=recurrent, mirror_state_idx=state_mirror_indices):

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
                    print(f"\t\tbatch had kl of {max(kls)} (threshold {kl_thresh}), stopping optimization early.")
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

    def _update_policy(self, states, actions, returns, advantages, mask, mirror_states=None, mirror_action_idx=None):
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

@ray.remote
class PPO_Sampler(PPO_Worker):
    """
    Worker for sampling experience for PPO

    Args:
        actor: actor pytorch network
        critic: critic pytorch network
        env_fn: environment constructor function
        gamma: discount factor


    Attributes:
        env: instance of environment
        gamma: discount factor
        dynamics_randomization: if dynamics_randomization is enabled in environment
    """
    def __init__(self, actor, critic, env_fn, gamma):
        self.gamma  = gamma
        self.env    = env_fn()

        if hasattr(self.env, 'dynamics_randomization'):
            self.dynamics_randomization = self.env.dynamics_randomization
        else:
            self.dynamics_randomization = False

        PPO_Worker.__init__(self, actor, critic)

    def collect_experience(self, max_traj_len, min_steps):
        """
        Function to sample experience

        Args:
            max_traj_len: maximum trajectory length of an episode
            min_steps: minimum total steps to sample
        """
        torch.set_num_threads(1)
        with torch.no_grad():
            start = time()

            num_steps = 0
            memory = Buffer(self.gamma)
            actor  = self.actor
            critic = self.critic

            while num_steps < min_steps:
                self.env.dynamics_randomization = self.dynamics_randomization
                state = torch.Tensor(self.env.reset())

                done = False
                value = 0
                traj_len = 0

                if hasattr(actor, 'init_hidden_state'):
                    actor.init_hidden_state()

                if hasattr(critic, 'init_hidden_state'):
                    critic.init_hidden_state()

                while not done and traj_len < max_traj_len:
                    state = torch.Tensor(state)
                    action = actor(state, deterministic=False)
                    value = critic(state)

                    next_state, reward, done, _ = self.env.step(action.numpy())

                    reward = np.array([reward])

                    memory.push(state.numpy(), action.numpy(), reward, value.numpy())

                    state = next_state

                    traj_len += 1
                    num_steps += 1

                value = (not done) * critic(torch.Tensor(state)).numpy()
                memory.end_trajectory(terminal_value=value)

        return memory

    def evaluate(self, trajs=1, max_traj_len=400):
        """
        Function to evaluate

        Args:
            max_traj_len: maximum trajectory length of an episode
            trajs: minimum trajectories to evaluate for
        """
        with torch.no_grad():
            ep_returns = []
            traj_lens = []
            for traj in range(trajs):
                self.env.dynamics_randomization = False
                state = torch.Tensor(self.env.reset())

                done = False
                traj_len = 0
                ep_return = 0

                if hasattr(self.actor, 'init_hidden_state'):
                    self.actor.init_hidden_state()

                while not done and traj_len < max_traj_len:
                    action = self.actor(state, deterministic=True)

                    next_state, reward, done, _ = self.env.step(action.numpy())

                    state = torch.Tensor(next_state)
                    ep_return += reward
                    traj_len += 1
                ep_returns += [ep_return]
                traj_lens += [traj_len]

        return np.mean(ep_returns), np.mean(traj_lens)

class PPO(PPO_Worker):
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
        PPO_Worker.__init__(self, actor, critic)

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

        self.state_mirror_indices = self.env.get_state_mirror_indices() if hasattr(self.env, 'get_state_mirror_indices') else None
        self.action_mirror_indices = self.env.get_action_mirror_indices() if hasattr(self.env, 'get_action_mirror_indices') else None

        self.workers = [PPO_Sampler.remote(actor, critic, env_fn, args.discount) for _ in range(args.workers)]
        self.optim   = PPO_Optim.remote(actor, critic, **vars(args))

    def do_iteration(self, num_steps, max_traj_len, epochs, kl_thresh=0.02, verbose=True, batch_size=64, mirror=False):
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
        start = time()
        actor_param_id  = ray.put(list(self.actor.parameters()))
        critic_param_id = ray.put(list(self.critic.parameters()))
        norm_id = ray.put([self.actor.welford_state_mean, self.actor.welford_state_mean_diff, self.actor.welford_state_n])

        steps = max(num_steps // len(self.workers), max_traj_len)

        for w in self.workers:
            w.sync_policy.remote(actor_param_id, critic_param_id, input_norm=norm_id)

        if verbose:
            print("\t{:5.4f}s to copy policy params to workers.".format(time() - start))

        eval_rewards, eval_lens = zip(*ray.get([w.evaluate.remote(trajs=1, max_traj_len=max_traj_len) for w in self.workers]))
        eval_reward = np.mean(eval_rewards)
        avg_ep_len = np.mean(eval_lens)

        torch.set_num_threads(1)

        start   = time()
        buffers = ray.get([w.collect_experience.remote(max_traj_len, steps) for w in self.workers])
        memory  = merge_buffers(buffers)
        # Delete buffers to free up memory? Might not be necessary
        del buffers

        total_steps = len(memory)
        avg_batch_reward = np.mean(memory.ep_returns)
        elapsed     = time() - start
        sample_rate = (total_steps/1000)/elapsed
        if verbose:
            print(f"\t{elapsed:3.2f}s to collect {total_steps:6n} timesteps | {sample_rate:3.2}k/s.")

        if self.mirror > 0 and self.state_mirror_indices is not None and self.action_mirror_indices is not None:
            state_mirror_indices = self.state_mirror_indices
            action_mirror_indices = self.action_mirror_indices
        else:
            state_mirror_indices = None
            action_mirror_indices = None

        start  = time()
        done   = False

        update_time = time()
        self.optim.sync_policy.remote(actor_param_id, critic_param_id, input_norm=norm_id)
        losses = ray.get(self.optim.optimize.remote(ray.put(memory),
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
        update_time = time() - update_time
        sleep(0.25)
        if verbose:
            print(f"\t{update_time:3.2f}s to update policy.")
        return eval_reward, np.mean(kls), np.mean(a_loss), np.mean(c_loss), np.mean(m_loss), \
               len(memory), (sample_rate, update_time), total_steps, avg_ep_len, avg_batch_reward

def add_algo_args(parser):
    if isinstance(parser, argparse.ArgumentParser):
        parser.add_argument("--prenormalize_steps", default=100,           type=int)
        parser.add_argument("--num_steps",          default=500,          type=int)
        parser.add_argument('--discount',           default=0.99,          type=float)          # the discount factor
        parser.add_argument("--learn_stddev",       default=False,         action='store_true') # learn std_dev or keep it fixed
        parser.add_argument('--std',                default=0.13,          type=float)          # the fixed exploration std
        parser.add_argument("--a_lr",               default=1e-4,          type=float)          # adam learning rate for actor
        parser.add_argument("--c_lr",               default=1e-4,          type=float)          # adam learning rate for critic
        parser.add_argument("--eps",                default=1e-6,          type=float)          # adam eps
        parser.add_argument("--kl",                 default=0.02,          type=float)          # kl abort threshold
        parser.add_argument("--entropy_coeff",      default=0.0,           type=float)
        parser.add_argument("--clip",               default=0.2,           type=float)          # Clipping parameter for PPO surrogate loss
        parser.add_argument("--grad_clip",          default=0.05,          type=float)
        parser.add_argument("--batch_size",         default=16,            type=int)            # batch size for policy update
        parser.add_argument("--epochs",             default=1,             type=int)            # number of updates per iter
        parser.add_argument("--mirror",             default=0,             type=float)
        parser.add_argument("--do_prenorm",         default=False,         action='store_true') # Do pre-normalization or not

        parser.add_argument("--layers",             default="256,256",     type=str)            # hidden layer sizes in policy
        parser.add_argument("--arch",               default='ff')                               # either ff, lstm, or gru
        parser.add_argument("--bounded",            default=False,         type=bool)

        parser.add_argument("--workers",            default=10,             type=int)
        parser.add_argument("--redis",              default=None,          type=str)
        parser.add_argument("--previous",           default=None,          type=str)            # Dir of previously trained policy to start learning from
    elif isinstance(parser, SimpleNamespace) or isinstance(parser, argparse.Namespace()):
        default_values = {"prenormalize_steps" : 10,
                          "num_steps"          : 100,
                          "discount"           : 0.99,
                          "learn_stddev"       : False,
                          "std"                : 0.13,
                          "a_lr"               : 1e-4,
                          "c_lr"               : 1e-4,
                          "eps"                : 1e-6,
                          "kl"                 : 0.02,
                          "entropy_coeff"      : 0.0,
                          "clip"               : 0.2,
                          "grad_clip"          : 0.05,
                          "batch_size"         : 64,
                          "epochs"             : 3,
                          "mirror"             : 0,
                          "do_prenorm"         : False,
                          "layers"             : "256,256",
                          "arch"               : 'ff',
                          "bounded"            : False,
                          "workers"            : 2,
                          "redis"              : None,
                          "previous"           : None}

        for key, val in default_values.items():
            if not hasattr(parser, key):
                setattr(parser, key, val)

    return parser


def run_experiment(args, env_args):
    """
    Function to run a PPO experiment.

    Args:
        args: argparse namespace
    """
    from algo.util.normalization import train_normalizer
    from algo.util.log import create_logger
    from util.env_factory import env_factory

    from nn.critic import FFCritic, LSTMCritic, GRUCritic
    from nn.actor import FFActor, LSTMActor, GRUActor

    import locale
    locale.setlocale(locale.LC_ALL, '')

    # wrapper function for creating parallelized envs
    env_fn = env_factory(args.env_name, env_args)
    obs_dim = env_fn().observation_size
    action_dim = env_fn().action_size

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # NN loading
    std = torch.ones(action_dim)*args.std
    layers = [int(x) for x in args.layers.split(',')]
    if hasattr(args, "previous") and args.previous is not None:
        # TODO: copy optimizer states also???
        policy = torch.load(os.path.join(args.previous, "actor.pt"))
        critic = torch.load(os.path.join(args.previous, "critic.pt"))
        print("loaded model from {}".format(args.previous))
    else:
        if args.arch.lower() == 'lstm':
            policy = LSTMActor(obs_dim,
                               action_dim,
                               std=std,
                               bounded=False,
                               layers=layers,
                               learn_std=args.learn_stddev)
            critic = LSTMCritic(obs_dim, layers=layers)
        elif args.arch.lower() == 'gru':
            policy = GRUActor(obs_dim,
                              action_dim,
                              std=std,
                              bounded=False,
                              layers=layers,
                              learn_std=args.learn_stddev)
            critic = GRUCritic(obs_dim, layers=layers)
        elif args.arch.lower() == 'ff':
            policy = FFActor(obs_dim,
                             action_dim,
                             std=std,
                             bounded=False,
                             layers=layers,
                             learn_std=args.learn_stddev,
                             nonlinearity=torch.tanh)
            critic = FFCritic(obs_dim, layers=layers)
        else:
            raise RuntimeError(f"Archetecture {args.arch} is not included, check the entry point.")

        # Prenormalization
        if args.do_prenorm:
            print("Collecting normalization statistics with {} states...".format(args.prenormalize_steps))
            train_normalizer(env_fn, policy, args.prenormalize_steps, max_traj_len=args.traj_len, noise=1)
            critic.copy_normalizer_stats(policy)

    policy.train(True)
    critic.train(True)

    # create a tensorboard logging object
    logger = create_logger(args)
    args.save_actor_path = os.path.join(logger.dir, 'actor.pt')
    args.save_critic_path = os.path.join(logger.dir, 'critic.pt')
    args.save_path = logger.dir
    # create actor/critic dict tp include model_state_dict and other class attributes
    actor_dict = {}
    critic_dict = {}

    # Algo init
    algo = PPO(policy, critic, env_fn, args)

    print()
    print("Proximal Policy Optimization:")
    print("\tseed:               {}".format(args.seed))
    print("\tenv name:           {}".format(args.env_name))
    print("\tmirror:             {}".format(args.mirror))
    print("\ttimesteps:          {:n}".format(int(args.timesteps)))
    print("\titeration steps:    {:n}".format(int(args.num_steps)))
    print("\tprenormalize steps: {}".format(int(args.prenormalize_steps)))
    print("\ttraj_len:           {}".format(args.traj_len))
    print("\tdiscount:           {}".format(args.discount))
    print("\tactor_lr:           {}".format(args.a_lr))
    print("\tcritic_lr:          {}".format(args.c_lr))
    print("\tadam eps:           {}".format(args.eps))
    print("\tentropy coeff:      {}".format(args.entropy_coeff))
    print("\tgrad clip:          {}".format(args.grad_clip))
    print("\tbatch size:         {}".format(args.batch_size))
    print("\tepochs:             {}".format(args.epochs))
    print("\tworkers:            {}".format(args.workers))
    print()

    itr = 0
    timesteps = 0
    best_reward = None
    past500_reward = -1
    while timesteps < args.timesteps:
        eval_reward, kl, a_loss, c_loss, m_loss, steps, times, total_steps, avg_ep_len, avg_batch_reward = algo.do_iteration(args.num_steps, args.traj_len, args.epochs, batch_size=args.batch_size, kl_thresh=args.kl, mirror=args.mirror)

        timesteps += steps
        print(f"iter {itr:4d} | return: {eval_reward:5.2f} | KL {kl:5.4f} | Actor loss {a_loss:5.4f}" \
              f" | Critic loss {c_loss:5.4f} | ", end='')
        if m_loss != 0:
            print(f"mirror {m_loss:6.5f} | ", end='')

        print(f"timesteps {timesteps:n}")

        # Savhing checkpoints for best reward
        if best_reward is None or eval_reward > best_reward:
            print(f"\t(best policy so far! saving to {args.save_actor_path})")
            best_reward = eval_reward
            for key in vars(algo.actor):
                actor_dict[key] = getattr(algo.actor, key)
            for key in vars(algo.critic):
                critic_dict[key] = getattr(algo.critic, key)
            torch.save(actor_dict | {'model_state_dict': algo.actor.state_dict()},
                    args.save_actor_path)
            torch.save(critic_dict | {'model_state_dict': algo.critic.state_dict()},
                    args.save_critic_path)

        # Intermitent saving
        if itr % 500 == 0:
            past500_reward = -1
        if eval_reward > past500_reward:
            past500_reward = eval_reward
            for key in vars(algo.actor):
                actor_dict[key] = getattr(algo.actor, key)
            for key in vars(algo.critic):
                critic_dict[key] = getattr(algo.critic, key)
            torch.save(actor_dict | {'model_state_dict': algo.actor.state_dict()},
                       args.save_actor_path[:-3] + "_past500.pt")
            torch.save(critic_dict | {'model_state_dict': algo.critic.state_dict()},
                       args.save_critic_path[:-3] + "_past500.pt")

        if logger is not None:
            logger.add_scalar("Test/Return", eval_reward, itr)
            logger.add_scalar("Train/Return", avg_batch_reward, itr)
            logger.add_scalar("Train/Mean Eplen", avg_ep_len, itr)
            logger.add_scalar("Train/Mean KL Div", kl, itr)

            logger.add_scalar("Misc/Critic Loss", c_loss, itr)
            logger.add_scalar("Misc/Actor Loss", a_loss, itr)
            logger.add_scalar("Misc/Mirror Loss", m_loss, itr)
            logger.add_scalar("Misc/Timesteps", total_steps, itr)

            logger.add_scalar("Misc/Sample Times", times[0], itr)
            logger.add_scalar("Misc/Optimize Times", times[1], itr)

        itr += 1
    print(f"Finished ({timesteps} of {args.timesteps}).")

    if args.wandb:
        wandb.finish()
