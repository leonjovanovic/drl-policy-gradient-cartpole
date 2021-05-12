import torch
import torch.nn as nn

class PolicyNN(nn.Module):
    def __init__(self, input_shape, output_shape, seed):
        super(PolicyNN, self).__init__()
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        self.policy_nn = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_shape),
            nn.Softmax(dim=-1)
        )
        self.policy_nn.double()

    def forward(self, x):
        return self.policy_nn(x)
