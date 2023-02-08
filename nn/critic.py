import torch
import torch.nn as nn
import torch.nn.functional as F
import math as math

from nn.base import FF_Base, LSTM_Base, GRU_Base, CNN_Base, CNN_Depth, Mix_Base, Mix_FF_Base

class V:
    """
    The base class for Value functions.
    """
    def __init__(self, latent, env_name, perception_in=None, perception_out=None, perception_size=None):
        self.env_name          = env_name
        self.network_out       = nn.Linear(latent, 1)

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
            
    def v_forward(self, input_state, update=False):
        state = self.normalize_state(input_state, update=update)

        if self.perception:
            size = input_state.size()
            dims = len(size)
            
            # process inputs for CNN with iid data
            if dims == 3: # for optimizaton with batch of trajectories
                traj_len = size[0]
                robot_state = state[:,:,:-self.perception_in_size]
                cnn_feature_out = torch.empty(size=(size[0], size[1], self.perception_out_size)).to(input_state.device)
                for traj_idx in range(traj_len):
                    batch_data = state[traj_idx, :, -self.perception_in_size:]
                    perception_state = batch_data[-self.perception_in_size:].reshape(-1, self.image_channel, self.image_width, self.image_height)
                    cnn_feature = self.cnn_module.forward(perception_state).squeeze() # forward the CNN to get feature vector
                    cnn_feature_out[traj_idx,:,:] = cnn_feature
                state = torch.cat((robot_state, cnn_feature_out), dim=2) # concatenate feature vector with robot states

            elif dims == 1: # for model forward
                robot_state = state[:-self.perception_in_size]
                perception_state = state[-self.perception_in_size:].reshape(-1, 1, self.image_width, self.image_height)
                cnn_feature_out = self.cnn_module.cnn_net(perception_state).squeeze() # forward the CNN to get feature vector
                state = torch.cat((robot_state, cnn_feature_out)) # concatenate feature vector with robot states

        x = self._base_forward(state)
        return self.network_out(x)


class FF_V(FF_Base, V):
    """
    A class inheriting from FF_Base and V
    which implements a feedforward value function.
    """
    def __init__(self, input_dim, layers=(256, 256), env_name=None):
        FF_Base.__init__(self, input_dim, layers, F.relu)
        V.__init__(self, layers[-1], env_name)

    def forward(self, state):
        return self.v_forward(state)


class LSTM_V(LSTM_Base, V):
    """
    A class inheriting from LSTM_Base and V
    which implements a recurrent value function.
    """
    def __init__(self, input_dim, layers=(128, 128), env_name=None, perception_in=None, perception_out=None, perception_size=None):
        LSTM_Base.__init__(self, input_dim, layers)
        V.__init__(self, layers[-1], env_name, perception_in=perception_in, perception_out=perception_out, perception_size=perception_size)

        self.is_recurrent = True
        self.init_hidden_state()

    def forward(self, state):
        return self.v_forward(state)

class GRU_V(GRU_Base, V):
    """
    A class inheriting from GRU_Base and V
    which implements a recurrent value function.
    """
    def __init__(self, input_dim, layers=(128, 128), env_name=None):
        GRU_Base.__init__(self, input_dim, layers)
        V.__init__(self, layers[-1], env_name)

        self.is_recurrent = True
        self.init_hidden_state()

    def forward(self, state):
        return self.v_forward(state)

class Mix_Stochastic_V(Mix_Base, V):
    def __init__(self, input_dim, state_dim, nonstate_dim, action_dim, nonstate_encoder_dim, layers=(128, 128), env_name=None, bounded=False, fixed_std=None, perception_in=None, perception_out=None, perception_size=None, nonstate_encoder_on=True):
        Mix_Base.__init__(self, input_dim, state_dim, nonstate_dim, nonstate_encoder_dim, layers, layers, nonstate_encoder_on=nonstate_encoder_on)
        V.__init__(self, layers[-1], env_name, perception_in=perception_in, perception_out=perception_out, perception_size=perception_size)

        self.is_recurrent = True
        self.init_hidden_state()

    def forward(self, state):
        return self.v_forward(state)

class Mix_FF_Stochastic_V(Mix_FF_Base, V):
    def __init__(self, input_dim, state_dim, nonstate_dim, action_dim, nonstate_encoder_dim, layers=(128, 128), env_name=None, bounded=False, fixed_std=None, perception_in=None, perception_out=None, perception_size=None, nonstate_encoder_on=True):
        Mix_FF_Base.__init__(self, input_dim, state_dim, nonstate_dim, nonstate_encoder_dim, layers, layers, nonstate_encoder_on=nonstate_encoder_on)
        V.__init__(self, layers[-1], env_name, perception_in=perception_in, perception_out=perception_out, perception_size=perception_size)

        self.is_recurrent = False

    def forward(self, state):
        return self.v_forward(state)
