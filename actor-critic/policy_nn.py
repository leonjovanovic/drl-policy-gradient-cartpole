import torch
import torch.nn as nn

class PolicyNN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(PolicyNN, self).__init__()
        self.policy_nn = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, output_shape),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.policy_nn(x)
