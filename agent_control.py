from agent_nn import PolicyNN
import torch

class AgentControl:
    def __init__(self, env, hyperparameters):
        self.learning_rate = hyperparameters['learning_rate']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_nn = PolicyNN(env.observation_space.shape[0]).to(self.device)

    def select_action(self, obs):
        print(12)
        return self.policy_nn(torch.tensor(obs, dtype=torch.double).to(self.device))
