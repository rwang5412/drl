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

from util.mirror import mirror_tensor


class AlgoWorker:
    """
        Generic template for a worker (sampler or optimizer) for training algorithms

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