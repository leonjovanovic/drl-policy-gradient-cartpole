import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PolicyNN(nn.Module):
    def __init__(self, input_shape):
        super(PolicyNN, self).__init__()
        self.policy_nn = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.policy_nn.double()

    def forward(self, x):
        print(x.dtype)
        return self.policy_nn(x)
