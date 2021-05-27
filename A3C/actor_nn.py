import torch.nn as nn

class ActorNN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(ActorNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, output_shape),
            nn.Softmax(dim=-1)
        )
        self.model.double()

    def forward(self, x):
        return self.model(x)
