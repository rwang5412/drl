import torch

import torch.nn as nn
import math as math

from nn.base import FFBase, LSTMBase, GRUBase, MixBase

class Actor:
    def __init__(self, 
                 latent: int, 
                 action_dim: int, 
                 bounded: bool, 
                 learn_std: bool, 
                 std: float):
        """The base class for stochastic actors.

        Args:
            latent (int): Input size for last action layer.
            action_dim (int): Action dim for last action layer.
            bounded (bool): Additional tanh activation after last layer.
            learn_std (bool): Option to learn std.
            std (float): Constant std.
        """
        self.action_dim = action_dim
        self.bounded    = bounded
        self.std        = std
        self.means      = nn.Linear(latent, action_dim)
        self.learn_std  = learn_std
        if self.learn_std:
            self.log_stds = nn.Linear(latent, action_dim)

    def _get_distrbution_params(self, input_state, update_normalization_param):
        """Perform a complete forward pass of the model and output mean/std for policy
        forward in stochastic_forward()

        Args:
            input_state (_type_): Model input
            update (bool): Option to update prenorm params. Defaults to False.

        Returns:
            mu: Model output, ie, mean of the distribution
            std: Optionally trainable param for distribution std. Default is constant.
        """
        state = self.normalize_state(input_state, 
                                     update_normalization_param=update_normalization_param)
        latent = self._base_forward(state)
        mu = self.means(latent)
        if self.learn_std:
            std = torch.clamp(self.log_stds(latent), -2, 1).exp()
        else:
            std = self.std
        return mu, std

    def pdf(self, state):
        """Return Diagonal Normal Distribution object given mean/std from part of actor forward pass
        """
        mu, sd = self._get_distrbution_params(state, update_normalization_param=False)
        return torch.distributions.Normal(mu, sd)
    
    def log_prob(self, state, action):
        """Return the log probability of a distribution given state and action
        """
        log_prob = self.pdf(state=state).log_prob(action).sum(-1, keepdim=True)
        if self.bounded: # 
            log_prob -= torch.log((1 - torch.tanh(state).pow(2)) + 1e-6)
        return log_prob

    def actor_forward(self, 
                      state: torch.Tensor, 
                      deterministic=True, 
                      update_normalization_param=False):
        """Perform actor forward in either deterministic or stochastic way, ie, inference/training.
        This function is default to inference mode. 

        Args:
            state (torch.Tensor): Input to actor.
            deterministic (bool, optional): inference mode. Defaults to True.
            update_normalization_param (bool, optional): Toggle to update params. Defaults to False.

        Returns:
            action with or without noise.
        """
        mu, std = self._get_distrbution_params(state, 
                                               update_normalization_param=update_normalization_param)
        if not deterministic: # draw random samples for stochastic forward
            dist = torch.distributions.Normal(mu, std)
            sample = dist.rsample()
        if self.bounded:
            action = torch.tanh(mu) if deterministic else torch.tanh(sample)
        else:
            action = mu if deterministic else sample
        return action

class FFActor(FFBase, Actor):
    """
    A class inheriting from FF_Base and Actor
    which implements a feedforward stochastic policy.
    """
    def __init__(self, 
                 input_dim, 
                 action_dim, 
                 layers, 
                 bounded, 
                 learn_std,
                 std):
        FFBase.__init__(self, in_dim=input_dim, layers=layers)
        Actor.__init__(self, 
                       latent=layers[-1], 
                       action_dim=action_dim, 
                       bounded=bounded, 
                       learn_std=learn_std, 
                       std=std)

    def forward(self, x, deterministic=True, update_norm=False):
        return self.actor_forward(x, deterministic=deterministic, 
                                  update_normalization_param=update_norm)

class LSTMActor(LSTMBase, Actor):
    """
    A class inheriting from LSTM_Base and Actor
    which implements a recurrent stochastic policy.
    """
    def __init__(self, 
                 input_dim, 
                 action_dim, 
                 layers, 
                 bounded, 
                 learn_std,
                 std):

        LSTMBase.__init__(self, input_dim, layers)
        Actor.__init__(self, 
                       latent=layers[-1], 
                       action_dim=action_dim, 
                       bounded=bounded, 
                       learn_std=learn_std,
                       std=std)

        self.is_recurrent = True
        self.init_hidden_state()

    def forward(self, x, deterministic=True, update_norm=False):
        return self.actor_forward(x, deterministic=deterministic, 
                                  update_normalization_param=update_norm)

class MixActor(MixBase, Actor):
    """
    A class inheriting from Mix_Base and Actor
    which implements a recurrent + FF stochastic policy.
    """
    def __init__(self, 
                 input_dim, 
                 state_dim, 
                 nonstate_dim, 
                 action_dim, 
                 lstm_layers, 
                 ff_layers,
                 bounded, 
                 learn_std,
                 std,
                 nonstate_encoder_dim, 
                 nonstate_encoder_on):

        MixBase.__init__(self,
                          in_dim=input_dim, 
                          state_dim=state_dim, 
                          nonstate_dim=nonstate_dim, 
                          lstm_layers=lstm_layers, 
                          ff_layers=ff_layers, 
                          nonstate_encoder_dim=nonstate_encoder_dim,
                          nonstate_encoder_on=nonstate_encoder_on)
        Actor.__init__(self, 
                       latent=ff_layers[-1], 
                       action_dim=action_dim, 
                       bounded=bounded, 
                       learn_std=learn_std,
                       std=std)

        self.is_recurrent = True
        self.init_hidden_state()

    def forward(self, x, deterministic=True, update_norm=False):
        return self.actor_forward(x, deterministic=deterministic, 
                                  update_normalization_param=update_norm)

    def latent_space(self, x):
        return self._latent_space_forward(x)

class GRUActor(GRUBase, Actor):
    """
    A class inheriting from GRU_Base and Actor
    which implements a recurrent stochastic policy.
    """
    def __init__(self, 
                 input_dim, 
                 action_dim, 
                 layers, 
                 bounded, 
                 learn_std,
                 std):

        GRUBase.__init__(self, input_dim, layers)
        Actor.__init__(self, 
                       latent=layers[-1], 
                       action_dim=action_dim, 
                       bounded=bounded, 
                       learn_std=learn_std,
                       std=std)

        self.is_recurrent = True
        self.init_hidden_state()

    def forward(self, x, deterministic=True, update_norm=False):
        return self.actor_forward(x, deterministic=deterministic, 
                                  update_normalization_param=update_norm)
