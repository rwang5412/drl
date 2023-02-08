from typing import no_type_check_decorator
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import sqrt
from torch.nn.modules.activation import Sigmoid

def normc_fn(m):
    """
    This function multiplies the weights of a pytorch linear layer by a small
    number so that outputs early in training are close to zero, which means 
    that gradients are larger in magnitude. This means a richer gradient signal
    is propagated back and speeds up learning (probably).
    """
    if m.__class__.__name__.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

def create_layers(layer_fn, input_dim, layer_sizes):
    """
    This function creates a pytorch modulelist and appends
    pytorch modules like nn.Linear or nn.LSTMCell passed
    in through the layer_fn argument, using the sizes
    specified in the layer_sizes list.
    """
    ret = nn.ModuleList()
    ret += [layer_fn(input_dim, layer_sizes[0])]
    for i in range(len(layer_sizes)-1):
        ret += [layer_fn(layer_sizes[i], layer_sizes[i+1])]
    return ret

class Net(nn.Module):
    """
    The base class which all policy networks inherit from. It includes methods
    for normalizing states.
    """
    def __init__(self):
        super(Net, self).__init__()
        # nn.Module.__init__(self)
        self.is_recurrent = False

        self.welford_state_mean = torch.zeros(1)
        self.welford_state_mean_diff = torch.ones(1)
        self.welford_state_n = 1

        self.env_name = None

        self.calculate_norm = False

    def normalize_state(self, state, update=True):
        """
        Use Welford's algorithm to normalize a state, and optionally update the statistics
        for normalizing states using the new state, online.
        """
        #state = torch.Tensor(state)

        if self.welford_state_n == 1:
            self.welford_state_mean = torch.zeros(state.size(-1)).to(state.device)
            self.welford_state_mean_diff = torch.ones(state.size(-1)).to(state.device)

        if update:
            if len(state.size()) == 1:  # if we get a single state vector
                state_old = self.welford_state_mean
                self.welford_state_mean += (state - state_old) / self.welford_state_n
                self.welford_state_mean_diff += (state - state_old) * (state - state_old)
                self.welford_state_n += 1
            else:
                raise RuntimeError  # this really should not happen
        return (state - self.welford_state_mean) / sqrt(self.welford_state_mean_diff / self.welford_state_n)

    def copy_normalizer_stats(self, net):
        self.welford_state_mean      = net.welford_state_mean
        self.welford_state_mean_diff = net.welford_state_mean_diff
        self.welford_state_n         = net.welford_state_n
  
    def initialize_parameters(self):
        self.apply(normc_fn)
        if hasattr(self, 'network_out'):
            self.network_out.weight.data.mul_(0.01)

class FF_Base(Net):
    """
    The base class for feedforward networks.
    """
    def __init__(self, in_dim, layers, nonlinearity):
        super(FF_Base, self).__init__()
        self.layers       = create_layers(nn.Linear, in_dim, layers)
        self.nonlinearity = nonlinearity

    def _base_forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = self.nonlinearity(layer(x))
        return x

class LSTM_Base(Net):
    """
    The base class for LSTM networks.
    """
    def __init__(self, in_dim, layers):
        super(LSTM_Base, self).__init__()
        self.layers = create_layers(nn.LSTMCell, in_dim, layers)

    def init_hidden_state(self, batch_size=1):
        self.hidden = [torch.zeros(batch_size, l.hidden_size).to(next(self.layers.parameters()).device) for l in self.layers]
        self.cells  = [torch.zeros(batch_size, l.hidden_size).to(next(self.layers.parameters()).device) for l in self.layers]

    def get_hidden_state(self):
        hidden_numpy = [self.hidden[l].numpy() for l in range(len(self.layers))]
        cells_numpy = [self.cells[l].numpy() for l in range(len(self.layers))]

        return hidden_numpy, cells_numpy

    def set_hidden_state(self, hidden, cells):
        self.hidden = torch.FloatTensor(hidden)
        self.cells = torch.FloatTensor(cells)
        #self.hidden = [torch.FloatTensor(hidden_layer) for hidden_layer in hidden]
        #self.cells = [torch.FloatTensor(cells_layer) for cells_layer in cells]

    def _base_forward(self, x):
        dims = len(x.size())

        if dims == 3:  # if we get a batch of trajectories
            self.init_hidden_state(batch_size=x.size(1))

            y = []
            for t, x_t in enumerate(x):
                for idx, layer in enumerate(self.layers):
                    c, h = self.cells[idx], self.hidden[idx]
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]

                y.append(x_t)
            x = torch.stack([x_t for x_t in y])
        else:
            if dims == 1:  # if we get a single timestep (if not, assume we got a batch of single timesteps)
                x = x.view(1, -1)

            for idx, layer in enumerate(self.layers):
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]

            if dims == 1:
                x = x.view(-1)
        return x

class Mix_Base(Net):
    """
    the base class for mixing LSTM and FF for actor
    """
    def __init__(self, in_dim, state_dim, nonstate_dim, nonstate_encoder_dim, lstm_layers, ff_layers, nonlinearity=torch.nn.functional.relu, nonstate_encoder_on=True):
        assert state_dim+nonstate_dim==in_dim, "State and Nonstate Dimension Mismatch"
        super(Mix_Base, self).__init__()
        if nonstate_encoder_on: # use a FF encoder to encode commands
            nonstate_ft_dim = nonstate_encoder_dim # single layer encoder
            self.nonstate_encoder = create_layers(nn.Linear, input_dim=nonstate_dim, layer_sizes=[nonstate_ft_dim])
        else:
            nonstate_ft_dim = nonstate_dim
        self.lstm = create_layers(nn.LSTMCell, input_dim=state_dim, layer_sizes=lstm_layers)
        self.ff = create_layers(nn.Linear, input_dim=nonstate_ft_dim+lstm_layers[-1], layer_sizes=ff_layers)
        self.ts_predictor = create_layers(nn.Linear, input_dim=lstm_layers[-1], layer_sizes=(lstm_layers[-1],9))
        self.nonlinearity = nonlinearity
        self.state_dim = state_dim
        self.nonstate_encoder_on = nonstate_encoder_on

    def init_hidden_state(self, batch_size=1):
        self.hidden = [torch.zeros(batch_size, l.hidden_size).to(next(self.lstm.parameters()).device) for l in self.lstm]
        self.cells  = [torch.zeros(batch_size, l.hidden_size).to(next(self.lstm.parameters()).device) for l in self.lstm]

    def get_hidden_state(self):
        hidden_numpy = [self.hidden[l].numpy() for l in range(len(self.lstm))]
        cells_numpy = [self.cells[l].numpy() for l in range(len(self.lstm))]

        return hidden_numpy, cells_numpy

    def set_hidden_state(self, hidden, cells):
        self.hidden = torch.FloatTensor(hidden)
        self.cells = torch.FloatTensor(cells)

    def lstm_forward(self, x):
        dims = len(x.size())

        if dims == 3:  # if we get a batch of trajectories
            self.init_hidden_state(batch_size=x.size(1))

            y = []
            for t, x_t in enumerate(x):
                for idx, layer in enumerate(self.lstm):
                    c, h = self.cells[idx], self.hidden[idx]
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]

                y.append(x_t)
            x = torch.stack([x_t for x_t in y])
        else:
            if dims == 1:  # if we get a single timestep (if not, assume we got a batch of single timesteps)
                x = x.view(1, -1)

            for idx, layer in enumerate(self.lstm):
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]

            if dims == 1:
                x = x.view(-1)
        return x

    def ff_forward(self, x):
        for idx, layer in enumerate(self.ff):
            x = self.nonlinearity(layer(x))
        return x

    def nonstate_encoder_forward(self, x):
        for idx, layer in enumerate(self.nonstate_encoder):
            x = self.nonlinearity(layer(x))
        return x

    def _base_forward(self, x):
        size = x.size()
        dims = len(size)
        if dims == 3: # for optimizaton with batch of trajectories
            state = x[:,:,:self.state_dim]
            nonstate = x[:,:,self.state_dim:]
            lstm_feature = self.lstm_forward(state)
            if self.nonstate_encoder_on:
                nonstate = self.nonstate_encoder_forward(nonstate)
            ff_input = torch.cat((lstm_feature, nonstate), dim=2)
        elif dims == 1: # for model forward
            state = x[:self.state_dim]
            nonstate = x[self.state_dim:]
            lstm_feature = self.lstm_forward(state)
            if self.nonstate_encoder_on:
                nonstate = self.nonstate_encoder_forward(nonstate)
            ff_input = torch.cat((lstm_feature, nonstate))
        ff_feature = self.ff_forward(ff_input)
        return ff_feature

    def _base_predict_ts_forward(self, x):
        size = x.size()
        dims = len(size)
        if dims == 3: # for optimizaton with batch of trajectories
            state = x[:,:,:self.state_dim]
            lstm_feature = self.lstm_forward(state)
        elif dims == 1: # for model forward
            state = x[:self.state_dim]
            lstm_feature = self.lstm_forward(state)
        for idx, layer in enumerate(self.ts_predictor):
            x = self.nonlinearity(layer(lstm_feature))
        return x
class Mix_FF_Base(Net):
    """
    the base class for mixing actor, dynamcis layer is also FF
    """
    def __init__(self, in_dim, state_dim, nonstate_dim, nonstate_encoder_dim, lstm_layers, ff_layers, nonlinearity=torch.nn.functional.relu, nonstate_encoder_on=True):
        assert state_dim+nonstate_dim==in_dim, "State and Nonstate Dimension Mismatch"
        super(Mix_FF_Base, self).__init__()
        if nonstate_encoder_on: # use a FF encoder to encode commands
            nonstate_ft_dim = nonstate_encoder_dim
            self.nonstate_encoder = create_layers(nn.Linear, input_dim=nonstate_dim, layer_sizes=(nonstate_dim, nonstate_ft_dim))
        else:
            nonstate_ft_dim = nonstate_dim
        self.dynamics = create_layers(nn.Linear, input_dim=state_dim, layer_sizes=lstm_layers)
        self.ff = create_layers(nn.Linear, input_dim=nonstate_ft_dim+lstm_layers[-1], layer_sizes=ff_layers)
        self.nonlinearity = nonlinearity
        self.state_dim = state_dim
        self.nonstate_encoder_on = nonstate_encoder_on

    def ff_forward(self, x):
        for idx, layer in enumerate(self.ff):
            x = self.nonlinearity(layer(x))
        return x

    def dyn_forward(self, x):
        for idx, layer in enumerate(self.dynamics):
            x = self.nonlinearity(layer(x))
        return x

    def nonstate_encoder_forward(self, x):
        for idx, layer in enumerate(self.nonstate_encoder):
            x = self.nonlinearity(layer(x))
        return x

    def _base_forward(self, x):
        size = x.size()
        dims = len(size) # dim of the x
        if dims == 3: # batch of trajectories [traj, batch, iid_data]
            state = x[:,:,:self.state_dim]
            nonstate = x[:,:,self.state_dim:]
            dyn_feature = self.dyn_forward(state)
            if self.nonstate_encoder_on:
                nonstate = self.nonstate_encoder_forward(nonstate)
            ff_input = torch.cat((dyn_feature, nonstate), dim=2)
        elif dims == 2: # for iid training [batch_size, iid_data_size]
            state = x[:, :self.state_dim]
            nonstate = x[:, self.state_dim:]
            dyn_feature = self.dyn_forward(state)
            if self.nonstate_encoder_on:
                nonstate = self.nonstate_encoder_forward(nonstate)
            ff_input = torch.cat((dyn_feature, nonstate), dim=1)
        elif dims == 1: # for model forward [iid_data]
            state = x[:self.state_dim]
            nonstate = x[self.state_dim:]
            dyn_feature = self.dyn_forward(state)
            if self.nonstate_encoder_on:
                nonstate = self.nonstate_encoder_forward(nonstate)
            ff_input = torch.cat((dyn_feature, nonstate))
        ff_feature = self.ff_forward(ff_input)
        return ff_feature

class GRU_Base(Net):
    """
    The base class for GRU networks.
    """
    def __init__(self, in_dim, layers):
        super(GRU_Base, self).__init__()
        self.layers = create_layers(nn.GRUCell, in_dim, layers)

    def init_hidden_state(self, batch_size=1):
        self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.layers]

    def _base_forward(self, x):
        dims = len(x.size())

        if dims == 3:  # if we get a batch of trajectories
            self.init_hidden_state(batch_size=x.size(1))

            y = []
            for t, x_t in enumerate(x):
                for idx, layer in enumerate(self.layers):
                    h = self.hidden[idx]
                    self.hidden[idx] = layer(x_t, h)
                    x_t = self.hidden[idx]
                y.append(x_t)
            x = torch.stack([x_t for x_t in y])
        else:
            if dims == 1:  # if we get a single timestep (if not, assume we got a batch of single timesteps)
                x = x.view(1, -1)

            for idx, layer in enumerate(self.layers):
                h = self.hidden[idx]
                self.hidden[idx] = layer(x, h)
                x = self.hidden[idx]

            if dims == 1:
                x = x.view(-1)
        return x

class CNN_Encoder(nn.Module):
    def __init__(self):
        super(CNN_Encoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1,4,3,2,1), #bs , 4 , 10 , 10
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, 2, 1), #bs , 8 5 5
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 5, 1, 0), # bs 16 1 1
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class CNN_Decoder(nn.Module):
    def __init__(self):
        super(CNN_Decoder, self).__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 5, 1, 0), #bs , 8 5 5
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1), #bs , 4 , 10 , 10
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=3, stride=2, padding=1, output_padding=1), #bs , 1 , 20 , 20
            #nn.BatchNorm2d(1),
            nn.Sigmoid()

        )

    def forward(self, x):
        x = self.deconv(x)
        return x  


class CNN_AE(nn.Module):

    def __init__(self):
        super(CNN_AE, self).__init__()
        self.encoder = CNN_Encoder()
        self.decoder = CNN_Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class CNN_Base(nn.Module):
    def __init__(self, in_dim, out_dim, image_size, 
                conv_kernel_size=5, stride=1, padding=0, maxpool=2):
        super(CNN_Base, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, conv_kernel_size, stride, padding) # 1 in channel, 4 out channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(4, 8, conv_kernel_size, stride, padding) # 4 in channel, 8 out channels, 5x5 square convolution kernel
        self.maxpool = nn.MaxPool2d(maxpool) # window size =(2, 2)
        self.fc1 = nn.Linear(8 * 4 * 4, 64)  # 2*2 from image dimension 
        self.fc2 = nn.Linear(64, out_dim)
        self.init_network()
    
    def init_network(self):
        self.cnn_net = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.maxpool,
            self.conv2,
            nn.ReLU(),
            nn.Flatten(),
            self.fc1,
            nn.ReLU(),
            self.fc2,
        )

    def forward(self, x):
        return self.cnn_net(x)

class CNN_Depth0(nn.Module):
    def __init__(self, in_dim, out_dim, image_size, 
                conv_kernel_size=31, stride=1, padding=0, maxpool=2):
        super(CNN_Depth, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, conv_kernel_size, stride, padding) # 1 in channel, 4 out channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(4, 8, conv_kernel_size, stride, padding) # 4 in channel, 8 out channels, 5x5 square convolution kernel
        self.maxpool = nn.MaxPool2d(maxpool) # window size =(2, 2)
        self.fc1 = nn.Linear(952, 128)  # 2*2 from image dimension 
        self.fc2 = nn.Linear(128, out_dim)
        self.init_network()
    
    def init_network(self):
        self.cnn_net = nn.Sequential(
            self.conv1, # w=130, h=90
            nn.ReLU(),
            self.maxpool, # 65, 45
            self.conv2, # 35, 15
            nn.ReLU(),
            self.maxpool, # 17, 7
            nn.Flatten(),
            self.fc1,
            nn.ReLU(),
            self.fc2,
        )

    def forward(self, x):
        return self.cnn_net(x)

class CNN_Depth(nn.Module):
    def __init__(self, in_dim=0, out_dim=0, image_size=0):
        super(CNN_Depth, self).__init__()
    
        self.cnn_net = nn.Sequential(
            nn.Conv2d(1,4,7,2,0), #bs , 4 , 10 , 10
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, 5, 2, 0), #bs , 8 5 5
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 1, 0), # bs 16 1 1
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 0), 
            nn.BatchNorm2d(16),
            nn.ReLU() 

        )
    
    def forward(self, x):
        x = self.cnn_net(x)
        return x

class StepPredictionModel(nn.Module):
    def __init__(self):
        super(StepPredictionModel, self).__init__()
        self.linear1 = nn.Linear(42+2, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)
        self.init_net()
    
    def init_net(self):
        self.net = nn.Sequential(self.linear1, 
                                nn.ReLU(),
                                self.linear2,
                                nn.ReLU(),
                                self.linear3,
                                nn.Sigmoid())

    def forward(self, x):
        return self.net(x)

class StepPrediction2DModel(nn.Module):
    def __init__(self):
        super(StepPrediction2DModel, self).__init__()
        self.deconv2d1 = nn.ConvTranspose2d(
            in_channels=42, out_channels=16, kernel_size=3, stride=2)
        self.deconv2d2 = nn.ConvTranspose2d(
            in_channels=16, out_channels=8, kernel_size=5, stride=2)
        self.deconv2d3 = nn.ConvTranspose2d(
            in_channels=8, out_channels=4, kernel_size=7, stride=2)
        self.deconv2d4 = nn.ConvTranspose2d(
            in_channels=4, out_channels=1, kernel_size=8, stride=1)    
        self.init_net()

    def init_net(self):
        self.net = nn.Sequential(self.deconv2d1,
                                 nn.ReLU(),
                                 self.deconv2d2,
                                 nn.ReLU(),
                                 self.deconv2d3,
                                 nn.ReLU(),
                                 self.deconv2d4,
                                 nn.Sigmoid()
                                 )

    def forward(self, x):
        return self.net(x)

class LSTM_Concat_CNN_Base(LSTM_Base):
    """
    A generic class that concat(output of CNN, raw robot states) as LSTM inputs
    CNN often needs redefine, so here constructs the CNN entirely than inheritance.
    Inputs: Concat array with the flattened image indexed in the end
    """
    def __init__(self, state_dim, layers, image_shape=(32,32), image_channel=1):
        self.perception_in_size  = int(image_shape[0] * image_shape[1])
        self.perception_out_size = 16 # make sure matches the CNN output size
        self.img_width   = image_shape[0]
        self.img_height  = image_shape[1]
        self.img_channel = image_channel
        LSTM_Base.__init__(self, in_dim=state_dim+self.perception_out_size, layers=layers)

        #NOTE: Taken from before, the following part will be modified per experiment
        self.cnn_net = nn.Sequential(
            nn.Conv2d(1,4,7,2,0), #bs , 4 , 10 , 10
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, 5, 2, 0), #bs , 8 5 5
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 1, 0), # bs 16 1 1
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 0), 
            nn.BatchNorm2d(16),
            nn.ReLU() 
        )

    def _base_forward(self, state):
        size = state.size()
        dims = len(size)

        if dims == 3: # for optimizaton with batch of trajectories
            traj_len = size[0]
            robot_state = state[:,:,:-self.perception_in_size]
            cnn_feature = []
            for traj_idx in range(traj_len):
                batch_data = state[traj_idx, :, -self.perception_in_size:]
                perception_state = batch_data[-self.perception_in_size:].reshape(-1, self.img_channel, self.img_width, self.img_height)
                cnn_feature.append(self.cnn_net.forward(perception_state).squeeze()) # forward the CNN to get feature vector
            cnn_feature_out = torch.stack([out for out in cnn_feature])
            x = torch.cat((robot_state, cnn_feature_out), dim=2) # concatenate feature vector with robot states
        elif dims == 1: # for model forward
            robot_state = state[:-self.perception_in_size]
            perception_state = state[-self.perception_in_size:].reshape(-1, self.img_channel, self.img_width, self.img_height)
            cnn_feature = self.cnn_net.forward(perception_state).squeeze() # forward the CNN to get feature vector
            x = torch.cat((robot_state, cnn_feature)) # concatenate feature vector with robot states

        return super()._base_forward(x)

# if __name__ == '__main__':
#     net = LSTM_Concat_CNN_Base(state_dim=30, layers=(32,32), image_shape=(32, 32))
#     print(net)
#     net.init_hidden_state()
#     net.eval() # call this for the batchnorm if single data
#     x = torch.ones((32*32+30))
#     # x = torch.ones((5, 8, 32*32+30))
#     print(net._base_forward(x).shape)