"""Proximal Policy Optimization (clip objective)."""
import argparse
import numpy as np
import os
import ray
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
                 **kwargs):
        AlgoWorker.__init__(self, actor, critic)
        self.old_actor = deepcopy(actor)
        self.actor_optim   = optim.Adam(self.actor.parameters(), lr=a_lr, eps=eps)
        self.critic_optim  = optim.Adam(self.critic.parameters(), lr=c_lr, eps=eps)
        self.entropy_coeff = entropy_coeff
        self.grad_clip = grad_clip
        self.mirror    = mirror
        self.clip = clip

    def optimize(self, memory, epochs=4,
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

        self.state_mirror_indices = self.env.get_state_mirror_indices() if hasattr(self.env, 'get_state_mirror_indices') else None
        self.action_mirror_indices = self.env.get_action_mirror_indices() if hasattr(self.env, 'get_action_mirror_indices') else None

        self.workers = [AlgoSampler.remote(actor, critic, env_fn, args.discount) for _ in range(args.workers)]
        self.optim   = PPOOptim.remote(actor, critic, **vars(args))

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
        memory = buffers[0]
        for i in range(1, len(buffers)):
            memory += buffers[i]
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
        parser.add_argument("--num_steps",          default=5000,          type=int)
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
        parser.add_argument("--batch_size",         default=64,            type=int)            # batch size for policy update
        parser.add_argument("--epochs",             default=3,             type=int)            # number of updates per iter
        parser.add_argument("--mirror",             default=0,             type=float)
        parser.add_argument("--do_prenorm",         default=False,         action='store_true') # Do pre-normalization or not

        parser.add_argument("--layers",             default="256,256",     type=str)            # hidden layer sizes in policy
        parser.add_argument("--arch",               default='ff')                               # either ff, lstm, or gru
        parser.add_argument("--bounded",            default=False,         type=bool)

        parser.add_argument("--workers",            default=2,             type=int)
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
    # env_fn = env_factory(**vars(args))
    env_fn = env_factory(args.env_name, env_args)

    obs_dim = env_fn().observation_size
    action_dim = env_fn().action_size

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
            raise RuntimeError

        # Prenormalization
        if args.do_prenorm:
            print("Collecting normalization statistics with {} states...".format(args.prenormalize_steps))
            train_normalizer(env_fn, policy, args.prenormalize_steps, max_traj_len=args.traj_len, noise=1)
            critic.copy_normalizer_stats(policy)
        else:
            policy.obs_mean = torch.zeros(obs_dim)
            policy.obs_std = torch.ones(obs_dim)
            critic.obs_mean = policy.obs_mean
            critic.obs_std = policy.obs_std

    policy.train(True)
    critic.train(True)

    if args.wandb:
        import wandb
        wandb.init(group = args.run_name, project=args.wandb_project_name, config=args, sync_tensorboard=True)

    algo = PPO(policy, critic, env_fn, args)

    # create a tensorboard logging object
    if not args.nolog:
        logger = create_logger(args)
    else:
        logger = None

    if not args.nolog:
        args.save_actor = os.path.join(logger.dir, 'actor.pt')
        args.save_critic = os.path.join(logger.dir, 'critic.pt')

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

        if not args.nolog and (best_reward is None or eval_reward > best_reward):
            print(f"\t(best policy so far! saving to {args.save_actor})")
            best_reward = eval_reward
            if args.save_actor is not None:
                torch.save(algo.actor, args.save_actor)

            if args.save_critic is not None:
                torch.save(algo.critic, args.save_critic)

        if itr % 500 == 0:
            past500_reward = -1
        if eval_reward > past500_reward:
            past500_reward = eval_reward
            if not args.nolog and args.save_actor is not None:
                torch.save(algo.actor, args.save_actor[:-4] + "_past500.pt")

            if not args.nolog and args.save_critic is not None:
                torch.save(algo.critic, args.save_critic[:-4] + "_past500.pt")

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
        wandb.join()
