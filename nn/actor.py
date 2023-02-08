import torch
import torch.nn as nn
import torch.nn.functional as F
import math as math

from nn.base import FF_Base, LSTM_Base, GRU_Base, CNN_Base, CNN_AE, CNN_Encoder, CNN_Decoder, CNN_Depth, Mix_Base, Mix_FF_Base
from nn.base import LSTM_Concat_CNN_Base

class Stochastic_Actor:
    """
    The base class for stochastic actors.
    """
    def __init__(self, latent, action_dim, env_name, bounded, fixed_std=None, perception_in=None, perception_out=None, perception_size=None):

        self.action_dim        = action_dim
        self.env_name          = env_name
        self.means             = nn.Linear(latent, action_dim)
        self.bounded           = bounded

        if perception_in is not None and perception_out is not None:
            self.perception = True
            self.perception_in_size  = perception_in
            self.perception_out_size = perception_out
            self.image_width = perception_size[0]
            self.image_height = perception_size[1]
            self.image_channel = 1
            if perception_in < 1000: # TODO: better robust handling of network
                self.cnn_module = CNN_Base(in_dim=perception_in, out_dim=perception_out, image_size=perception_size)
            else:
                self.cnn_module = CNN_Depth(in_dim=perception_in, out_dim=perception_out, image_size=perception_size)
        else:
            self.perception = False

        if fixed_std is None:
            self.log_stds = nn.Linear(latent, action_dim)
        self.fixed_std = fixed_std

    def _get_dist_params(self, input_state, update=False):
        state = self.normalize_state(input_state, update=update) # this won't affect since we dont use prenorm anymore
        size = state.size()
        dims = len(size)
        # print("input size", state.size())
        # print("sample dim", dims)

        if self.perception: # process inputs for CNN with iid data
            if dims == 3: # for optimizaton with batch of trajectories
                traj_len = size[0]
                robot_state = state[:,:,:-self.perception_in_size]
                cnn_feature_out = torch.empty(size=(size[0], size[1], self.perception_out_size)).to(input_state.device) # preallocate torch tensors
                # print("robot state", robot_state.shape)
                # print("perception state", cnn_feature_out.shape)
                for traj_idx in range(traj_len):
                    batch_data = state[traj_idx, :, -self.perception_in_size:]
                    # print("each traj size", batch_data.shape)
                    perception_state = batch_data[-self.perception_in_size:].reshape(-1, self.image_channel, self.image_width, self.image_height)
                    # print("cnn input size", perception_state.shape)
                    cnn_feature = self.cnn_module.forward(perception_state).squeeze() # forward the CNN to get feature vector
                    # print("iid feature size", cnn_feature.shape)
                    cnn_feature_out[traj_idx,:,:] = cnn_feature

                # print("cnn feature traj", cnn_feature_out.shape)
                # input()
                state = torch.cat((robot_state, cnn_feature_out), dim=2) # concatenate feature vector with robot states

            elif dims == 1: # for model forward
                robot_state = state[:-self.perception_in_size]
                # perception_state = state[-self.perception_in_size:].reshape(-1, self.image_channel, self.image_width, self.image_height)
                perception_state = state[-self.perception_in_size:].reshape(-1, self.image_channel, 32, 32)
                cnn_feature_out = self.cnn_module.cnn_net(perception_state).squeeze() # forward the CNN to get feature vector
                state = torch.cat((robot_state, cnn_feature_out)) # concatenate feature vector with robot states

            # print("ok to cat") if input_state.size() ==3 else None
            # input() if input_state.size() ==3 else None

        x = self._base_forward(state)
        # print("ok to forward") if input_state.size() ==3 else None
        # input() if input_state.size() ==3 else None

        mu = self.means(x)

        if self.fixed_std is None:
            std = torch.clamp(self.log_stds(x), -2, 1).exp().to(mu.device)
        else:
            std = self.fixed_std.to(mu.device)

        return mu, std

    def stochastic_forward(self, state, deterministic=True, update=False, log_probs=False):
        mu, sd = self._get_dist_params(state, update=update)

        if not deterministic or log_probs:
            dist = torch.distributions.Normal(mu, sd)
            sample = dist.rsample()

        if self.bounded:
            action = torch.tanh(mu) if deterministic else torch.tanh(sample)
        else:
            action = mu if deterministic else sample

        if log_probs:
            log_prob = dist.log_prob(sample)
            if self.bounded:
                log_prob -= torch.log((1 - torch.tanh(sample).pow(2)) + 1e-6)

            return action, log_prob.sum(1, keepdim=True)
        else:
            return action

    def pdf(self, state):
        mu, sd = self._get_dist_params(state)
        return torch.distributions.Normal(mu, sd)


class FF_Stochastic_Actor(FF_Base, Stochastic_Actor):
    """
    A class inheriting from FF_Base and Stochastic_Actor
    which implements a feedforward stochastic policy.
    """
    def __init__(self, input_dim, action_dim, layers=(256, 256), env_name=None, nonlinearity=torch.tanh, bounded=False, fixed_std=None):
        FF_Base.__init__(self, input_dim, layers, nonlinearity)
        Stochastic_Actor.__init__(self, layers[-1], action_dim, env_name, bounded, fixed_std=fixed_std)

    def forward(self, x, deterministic=True, update_norm=False, return_log_probs=False):
        return self.stochastic_forward(x, deterministic=deterministic, update=update_norm, log_probs=return_log_probs)

class LSTM_Stochastic_Actor(LSTM_Base, Stochastic_Actor):
    """
    A class inheriting from LSTM_Base and Stochastic_Actor
    which implements a recurrent stochastic policy.
    """
    def __init__(self, input_dim, action_dim, layers=(128, 128), env_name=None, bounded=False, fixed_std=None, perception_in=None, perception_out=None, perception_size=None):

        LSTM_Base.__init__(self, input_dim, layers)
        Stochastic_Actor.__init__(self, layers[-1], action_dim, env_name, bounded, fixed_std=fixed_std, perception_in=perception_in, perception_out=perception_out, perception_size=perception_size)

        self.is_recurrent = True
        self.init_hidden_state()

    def forward(self, x, deterministic=True, update_norm=False, return_log_probs=False):
        return self.stochastic_forward(x, deterministic=deterministic, update=update_norm, log_probs=return_log_probs)

class Mix_Stochastic_Actor(Mix_Base, Stochastic_Actor):
    """
    A class inheriting from Mix_Base and Stochastic_Actor
    which implements a recurrent + FF stochastic policy.
    """
    def __init__(self, input_dim, state_dim, nonstate_dim, action_dim, nonstate_encoder_dim, \
        layers=(128, 128), env_name=None, bounded=False, fixed_std=None, perception_in=None, perception_out=None, perception_size=None, \
            nonstate_encoder_on=True):

        Mix_Base.__init__(self, input_dim, state_dim, nonstate_dim, nonstate_encoder_dim, layers, layers, nonstate_encoder_on=nonstate_encoder_on)
        Stochastic_Actor.__init__(self, layers[-1], action_dim, env_name, bounded, fixed_std=fixed_std, perception_in=perception_in, perception_out=perception_out, perception_size=perception_size)

        self.is_recurrent = True
        self.init_hidden_state()

    def forward(self, x, deterministic=True, update_norm=False, return_log_probs=False):
        return self.stochastic_forward(x, deterministic=deterministic, update=update_norm, log_probs=return_log_probs)

    def predict_ts(self, x):
        return self._base_predict_ts_forward(x)

class Mix_FF_Stochastic_Actor(Mix_FF_Base, Stochastic_Actor):
    """
    A class inheriting from Mix_Base and Stochastic_Actor
    which implements a FF dynamics + FF stochastic policy.
    """
    def __init__(self, input_dim, state_dim, nonstate_dim, action_dim, nonstate_encoder_dim, \
        layers=(128, 128), env_name=None, bounded=False, fixed_std=None, perception_in=None, perception_out=None, perception_size=None, \
            nonstate_encoder_on=True):

        Mix_FF_Base.__init__(self, input_dim, state_dim, nonstate_dim, nonstate_encoder_dim, layers, layers, nonstate_encoder_on=nonstate_encoder_on)
        Stochastic_Actor.__init__(self, layers[-1], action_dim, env_name, bounded, fixed_std=fixed_std, perception_in=perception_in, perception_out=perception_out, perception_size=perception_size)

        self.is_recurrent = False

    def forward(self, x, deterministic=True, update_norm=False, return_log_probs=False):
        return self.stochastic_forward(x, deterministic=deterministic, update=update_norm, log_probs=return_log_probs)

class GRU_Stochastic_Actor(GRU_Base, Stochastic_Actor):
    """
    A class inheriting from GRU_Base and Stochastic_Actor
    which implements a recurrent stochastic policy.
    """
    def __init__(self, input_dim, action_dim, layers=(128, 128), env_name=None, bounded=False, fixed_std=None):

        GRU_Base.__init__(self, input_dim, layers)
        Stochastic_Actor.__init__(self, layers[-1], action_dim, env_name, bounded, fixed_std=fixed_std)

        self.is_recurrent = True
        self.init_hidden_state()

    def forward(self, x, deterministic=True, update_norm=False, return_log_probs=False):
        return self.stochastic_forward(x, deterministic=deterministic, update=update_norm, log_probs=return_log_probs)
        
class Jac_Actor:
    """
    The base class for jacobian-based exploration actors. Using multivariantNormal().
    """
    def __init__(self, latent, action_dim, env_name, bounded, fixed_std=None, perception_in=None, perception_out=None, perception_size=None):

        self.action_dim        = action_dim
        self.env_name          = env_name
        self.means             = nn.Linear(latent, action_dim)
        self.bounded           = bounded

        if fixed_std is None:
            self.log_stds = nn.Linear(latent, action_dim)
        
        ts_noise_up = torch.Tensor([[0,         0, fixed_std, fixed_std, 0],
                                    [fixed_std, 0, 0.0      , 0.0     ,  0],
                                    [0,         0, fixed_std, fixed_std, 0]])

        self.fixed_std = torch.hstack((torch.vstack((ts_noise_up,        torch.zeros((3,5)))), 
                                (torch.vstack((torch.zeros((3,5)), ts_noise_up)))))

    def _get_dist_params(self, input_state, update=False, jac_pinv=None):
        state = self.normalize_state(input_state, update=update) # this won't affect since we dont use prenorm anymore
        x = self._base_forward(state)
        mu = self.means(x)
        if self.fixed_std is None:
            std = torch.clamp(self.log_stds(x), -2, 1).exp()
        else:
            if jac_pinv is not None:
                std = torch.matmul(jac_pinv, self.fixed_std) # 10x6 x 6x10 = 10x10 gaussian
                std[1,1] = 0.13
                std[4,4] = 0.13
                std[6,6] = 0.13
                std[9,9] = 0.13
            else:
                raise Exception("must provide jacobian for noise sampling.")

        return mu, std

    def stochastic_forward(self, state, deterministic=True, update=False, log_probs=False, jac_pinv=None):
        mu, sd = self._get_dist_params(state, update=update, jac_pinv=jac_pinv)

        if not deterministic or log_probs:
            dist = torch.distributions.MultivariateNormal(mu, sd)
            sample = dist.rsample()

        if self.bounded:
            action = torch.tanh(mu) if deterministic else torch.tanh(sample)
        else:
            action = mu if deterministic else sample

        if log_probs:
            log_prob = dist.log_prob(sample)
            if self.bounded:
                log_prob -= torch.log((1 - torch.tanh(sample).pow(2)) + 1e-6)

            return action, log_prob.sum(1, keepdim=True)
        else:
            return action

    def pdf(self, state):
        mu, sd = self._get_dist_params(state)
        return torch.distributions.Normal(mu, sd)

class Stochastic_Actor_2:
    """
    The base class for stochastic actors.
    """
    def __init__(self, latent, action_dim, bounded, fixed_std=None):

        self.action_dim        = action_dim
        self.means             = nn.Linear(latent, action_dim)
        self.bounded           = bounded

        if fixed_std is None:
            self.log_stds = nn.Linear(latent, action_dim)
        self.fixed_std = fixed_std

    def _get_dist_params(self, input_state, update=False):
        state = self.normalize_state(input_state, update=update) # this won't affect since we dont use prenorm anymore
        x = self._base_forward(state)
        mu = self.means(x)

        if self.fixed_std is None:
            std = torch.clamp(self.log_stds(x), -2, 1).exp()
        else:
            std = self.fixed_std

        return mu, std

    def stochastic_forward(self, state, deterministic=True, update=False, log_probs=False):
        mu, sd = self._get_dist_params(state, update=update)

        if not deterministic or log_probs:
            dist = torch.distributions.Normal(mu, sd)
            sample = dist.rsample()

        if self.bounded:
            action = torch.tanh(mu) if deterministic else torch.tanh(sample)
        else:
            action = mu if deterministic else sample

        if log_probs:
            log_prob = dist.log_prob(sample)
            if self.bounded:
                log_prob -= torch.log((1 - torch.tanh(sample).pow(2)) + 1e-6)

            return action, log_prob.sum(1, keepdim=True)
        else:
            return action

    def pdf(self, state):
        mu, sd = self._get_dist_params(state)
        return torch.distributions.Normal(mu, sd)

class LSTM_CNN_Actor(LSTM_Concat_CNN_Base, Stochastic_Actor_2):
    """
    A class inheriting from LSTM_Base and Stochastic_Actor
    which implements a recurrent stochastic policy.
    """
    def __init__(self, input_dim, action_dim, layers=(128, 128), bounded=False, fixed_std=None, image_shape=None, image_channel=None):

        LSTM_Concat_CNN_Base.__init__(self, input_dim, layers, image_shape=image_shape, image_channel=image_channel)
        Stochastic_Actor_2.__init__(self, layers[-1], action_dim, bounded, fixed_std=fixed_std)

        self.is_recurrent = True
        self.init_hidden_state()

    def forward(self, x, deterministic=True, update_norm=False, return_log_probs=False):
        return self.stochastic_forward(x, deterministic=deterministic, update=update_norm, log_probs=return_log_probs)
