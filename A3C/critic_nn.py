import torch.nn as nn

class CriticNN(nn.Module):
    def __init__(self, input_shape, seed):
        super(CriticNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.model.double()

    def forward(self, x):
        return self.model(x)

