import numpy as np
import ray
import torch

from copy import deepcopy
from time import time, sleep
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence
from torch.nn.utils.rnn import pad_sequence

from algo.util.worker import AlgoWorker
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
    def __init__(self, discount: float = 0.99):
        self.discount = discount
        self.clear()

    def __len__(self):
        return self.size

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

    def push(self,
             state: np.ndarray,
             action: np.ndarray,
             reward: np.ndarray,
             value: np.ndarray,
             done: bool = False):
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

    def end_trajectory(self, terminal_value: float = 0.0):
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

    def _finish_buffer(self, state_mirror_idx = None):
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

    def sample(self,
               batch_size: int = 64,
               recurrent: bool = False,
               mirror_state_idx = None):
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

    def __add__(self, buf2):
        offset = len(self.states)

        new_buf = Buffer(self.discount)
        new_buf.states      = self.states + buf2.states
        new_buf.actions     = self.actions + buf2.actions
        new_buf.rewards     = self.rewards + buf2.rewards
        new_buf.values      = self.values + buf2.values
        new_buf.returns     = self.returns + buf2.returns
        new_buf.ep_returns  = self.ep_returns + buf2.ep_returns
        new_buf.ep_lens     = self.ep_lens + buf2.ep_lens

        new_buf.size        = self.size + buf2.size
        new_buf.traj_idx    = self.traj_idx + [offset + i for i in buf2.traj_idx[1:]]

        return new_buf

@ray.remote
class AlgoSampler(AlgoWorker):
    """
    Worker for sampling experience for training algorithms

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
    def __init__(self, actor, critic, env_fn, gamma, worker_id: int):
        self.gamma  = gamma
        self.env    = env_fn()
        self.worker_id = worker_id

        if hasattr(self.env, 'dynamics_randomization'):
            self.dynamics_randomization = self.env.dynamics_randomization
        else:
            self.dynamics_randomization = False

        AlgoWorker.__init__(self, actor, critic)

    def sample_traj(self, max_traj_len: int = 300, do_eval: bool = False, update_normalization_param: bool=False):
        """
        Function to sample experience

        Args:
            max_traj_len: maximum trajectory length of an episode
            min_steps: minimum total steps to sample
        """
        start_t = time()
        torch.set_num_threads(1)
        memory = Buffer(self.gamma)
        with torch.no_grad():
            state = torch.Tensor(self.env.reset())
            done = False
            value = 0
            traj_len = 0

            if hasattr(self.actor, 'init_hidden_state'):
                self.actor.init_hidden_state()
            if hasattr(self.critic, 'init_hidden_state'):
                self.critic.init_hidden_state()

            while not done and traj_len < max_traj_len:
                state = torch.Tensor(state)
                action = self.actor(state,
                                    deterministic=do_eval,
                                    update_normalization_param=update_normalization_param)
                if do_eval:
                    # If is evaluation, don't need critic value
                    value = 0.0
                else:
                    value = self.critic(state).numpy()
                next_state, reward, done, _ = self.env.step(action.numpy())
                reward = np.array([reward])

                memory.push(state.numpy(), action.numpy(), reward, value)
                state = next_state
                traj_len += 1

            value = (not done) * self.critic(torch.Tensor(state)).numpy()
            memory.end_trajectory(terminal_value=value)

        return memory, traj_len / (time() - start_t), self.worker_id
