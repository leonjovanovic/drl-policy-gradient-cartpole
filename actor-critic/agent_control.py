from critic_nn import CriticNN
from policy_nn import PolicyNN
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


class AgentControl:
    def __init__(self, env, hyperparameters):
        self.learning_rate_actor = hyperparameters['learning_rate_actor']
        self.learning_rate_critic = hyperparameters['learning_rate_critic']
        self.gamma = hyperparameters['gamma']
        self.seed = hyperparameters['random_seed']
        self.entropy_flag = hyperparameters['entropy']
        self.entropy = 0

        if self.seed != -1:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)

        self.input_shape = env.observation_space.shape[0]
        self.output_shape = env.action_space.n

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_nn = PolicyNN(self.input_shape, self.output_shape, self.seed).to(self.device)
        self.policy_optim = optim.Adam(params=self.policy_nn.parameters(), lr=self.learning_rate_actor)
        self.critic_nn = CriticNN(self.input_shape, self.seed).to(self.device)
        self.critic_optim = optim.Adam(params=self.critic_nn.parameters(), lr=self.learning_rate_critic)

        self.current_action_prob = torch.tensor(0)

    def choose_action(self, obs):
        action_prob = self.policy_nn(torch.tensor(obs, dtype=torch.float64).to(self.device))
        action = np.random.choice(np.array([0, 1]), p=action_prob.cpu().data.numpy())
        self.current_action_prob = action_prob[action]
        if self.entropy_flag:
            self.entropy = -torch.sum(action_prob * torch.log(action_prob)).detach()
        return action

    def update_critic_nn(self, reward, obs, new_obs):
        v_new = self.critic_nn(torch.tensor(new_obs, dtype=torch.float64).to(self.device)).detach()
        v_curr = self.critic_nn(torch.tensor(obs, dtype=torch.float64).to(self.device))
        mse = nn.MSELoss()
        loss = mse(reward + self.gamma * v_new, v_curr)
        if self.entropy_flag:
            loss += self.entropy

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        return loss.item()

    def evaluate_advantage(self, reward, obs, new_obs):
        v_new = self.critic_nn(torch.tensor(new_obs, dtype=torch.float64).to(self.device))
        v_curr = self.critic_nn(torch.tensor(obs, dtype=torch.float64).to(self.device))
        return reward + self.gamma * v_new - v_curr

    def update_actor_nn(self, advantage, obs):
        loss = -torch.log(self.current_action_prob) * advantage.detach()
        if self.entropy_flag:
            loss += self.entropy

        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()
        return loss.item()




