import torch
from torch import nn
import torch.nn.functional as F
import typing

class DuelingDQN(nn.Module):
    def __init__(self, channels: int, height: int, width: int, outputs: int):
        super(DuelingDQN, self).__init__()

        # parameters
        self._outputs = outputs
        self._channels = channels
        self._height = height
        self._width = width

        # CNN
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=6, stride=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Dense - Advantage stream
        convw = DuelingDQN._conv2d_size_out(DuelingDQN._conv2d_size_out(DuelingDQN._conv2d_size_out(width, kernel_size=6, stride=3),
                                    kernel_size=4, stride=2), kernel_size=3, stride=1)
        convh = DuelingDQN._conv2d_size_out(DuelingDQN._conv2d_size_out(DuelingDQN._conv2d_size_out(height, kernel_size=6, stride=3),
                                    kernel_size=4, stride=2), kernel_size=3, stride=1)
        linear_input_size = convw * convh * 64
        self.fc1_adv = nn.Linear(linear_input_size, 512)
        self.head_adv = nn.Linear(512, outputs)

        # Dense - Value stream
        self.fc1_val = nn.Linear(linear_input_size, 512)
        self.head_val = nn.Linear(512, 1)

    def _conv2d_size_out(size: int, kernel_size: int = 5, stride: int = 2):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)

        # Advantage stream
        x_adv = F.leaky_relu(self.fc1_adv(x))
        x_adv = F.softsign(self.head_adv(x_adv))

        # Value stream
        x_val = F.leaky_relu(self.fc1_val(x))
        x_val = self.head_val(x_val)

        # Combine the advantage and value streams to get Q-values
        x = x_val + x_adv - x_adv.mean(dim=1, keepdim=True)

        return x

# Functions (similar to the original ones)
# ...

def create_dueling(size: typing.List[int], outputs: int,
                   load_state_from: str = None, for_train=False) -> DuelingDQN:
    """
    create Dueling DQN model
    """
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
    provides best action for given state using Dueling DQN
    """
    hi = model(state).max(1)[1].view(1, 1).item()
    print(model(state), hi)
    return hi
