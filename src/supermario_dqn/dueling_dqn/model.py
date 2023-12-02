"""
Creates models for training and usage
"""

import torch
from torch import nn
# get this from dueling_dqn
import torch.nn.functional as F
import typing


class DuelingDQN(nn.Module):
    """
    class of the used Q-value Neural Network
    """

    def __init__(self, channels: int, height: int, width: int, outputs: int):
        super(DuelingDQN, self).__init__()

        # parameters (from dqn)
        self._outputs = outputs
        self._channels = channels
        self._height = height
        self._width = width

        # CNN (changed dueling to fit regular dqn params)
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(32 * height * width, outputs)
        self.q = nn.Linear(512, outputs)
        self.v = nn.Linear(512, 1)
                
        self.seq = nn.Sequential(self.conv1, self.conv2, self.fc, self.q, self.v)

        self.seq.apply(init_weights)

        # CNN (from Dueling)
        # device is for CPU or GPU
        # def __init__(self, n_frame, n_action, device):
        #     super(model, self).__init__()
        #     self.layer1 = nn.Conv2d(n_frame, 32, 8, 4)
        #     self.layer2 = nn.Conv2d(32, 64, 3, 1)
        #     self.fc = nn.Linear(20736, 512)
        #     self.q = nn.Linear(512, n_action)
        #     self.v = nn.Linear(512, 1)

        #     self.device = device
        #     self.seq = nn.Sequential(self.layer1, self.layer2, self.fc, self.q, self.v)

        #     self.seq.apply(init_weights)

        # Dense (from dueling)
        def forward(self, x):
            x = F.leaky_relu(self.bn1(self.conv1(x)))
            x = F.leaky_relu(self.bn2(self.conv2(x)))
            x = F.leaky_relu(self.bn3(self.conv3(x)))
            x = F.leaky_relu(self.fc1(x.view(x.size(0), -1)))
            x = F.softsign(self.head(x)) * 15.
            return x
    
        # Dense (from regular DQN)
        # convw = DQN._conv2d_size_out(DQN._conv2d_size_out(DQN._conv2d_size_out(width, kernel_size=6, stride=3),
        #                              kernel_size=4, stride=2), kernel_size=3, stride=1)
        # convh = DQN._conv2d_size_out(DQN._conv2d_size_out(DQN._conv2d_size_out(height, kernel_size=6, stride=3),
        #                              kernel_size=4, stride=2), kernel_size=3, stride=1)
        # linear_input_size = convw * convh * 64
        # self.fc1 = nn.Linear(linear_input_size, 512)
        # self.head = nn.Linear(512, outputs)

        # def _conv2d_size_out(size: int, kernel_size: int = 5, stride: int = 2):
        #     return (size - (kernel_size - 1) - 1) // stride + 1
    
        # Weights (from dueling)
        def init_weights(m):
            if type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

# FUNCTIONS

def create(size: typing.List[int], outputs: int, load_state_from: str = None, for_train=False) -> DuelingDQN:
    """
    create model
    """

    # equivalent to arrange(s) in dueling dqn
    if len(size) != 3 or size[0] < 0 or size[1] < 0 or size[2] < 0:
        raise ValueError(f'size must be positive: [channels, height, width]')

    dueling_dqn = DuelingDQN(size[0], size[1], size[2], outputs)

    if load_state_from is not None:
        if torch.cuda.is_available():
            dueling_dqn.load_state_dict(torch.load(load_state_from))
        else:
            dueling_dqn.load_state_dict(torch.load(load_state_from, map_location=torch.device('cpu')))

    if not for_train:
        dueling_dqn.eval()

    return dueling_dqn


def best_action(model: DuelingDQN, state: torch.Tensor) -> int:
    """
    provides best action for given state
    """
    return model(state).max(1)[1].view(1, 1).item()
