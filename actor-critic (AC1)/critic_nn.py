import torch
import torch.nn as nn

class CriticNN(nn.Module):
    def __init__(self, input_shape, seed):
        super(CriticNN, self).__init__()
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        self.critic_nn = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.critic_nn.double()

    def forward(self, x):
        return self.critic_nn(x)
