import torch
import torch.nn as nn
torch.manual_seed(48)
torch.cuda.manual_seed(48)

class CriticNN(nn.Module):
    def __init__(self, input_shape):
        super(CriticNN, self).__init__()
        self.critic_nn = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.critic_nn.double()

    def forward(self, x):
        return self.critic_nn(x)
