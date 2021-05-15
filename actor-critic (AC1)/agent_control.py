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
        self.entropy_beta = hyperparameters['entropy_beta']
        self.entropy = []

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
        self.loss = nn.MSELoss()

    def choose_action(self, obs):
        action_prob = self.policy_nn(torch.tensor(obs, dtype=torch.float64).to(self.device))
        action = np.random.choice(np.array([0, 1]), p=action_prob.cpu().data.numpy())
        if self.entropy_flag:
            self.entropy.append(-self.entropy_beta * torch.sum(action_prob * torch.log(action_prob)).detach())
        return action

    def calc_rewards_states(self, memory):
        i = len(memory) - 1
        v_new = self.critic_nn(torch.tensor(memory[i].new_obs, dtype=torch.float64).to(self.device)).detach()
        rewards = []
        states = []
        actions = []
        temp = v_new.item()
        while i > -1:
            states.append(memory[i].obs)
            actions.append(memory[i].action)
            rewards.append(memory[i].reward + self.gamma * temp)
            temp = memory[i].reward + self.gamma * temp
            i -= 1
        rewards = torch.tensor(rewards, dtype=torch.float64).to(self.device)
        states = torch.tensor(states, dtype=torch.float64).to(self.device)
        return rewards, states, actions

    def update_critic_nn(self, rewards, states):
        v_curr = self.critic_nn(states).squeeze(-1)
        loss = self.loss(rewards, v_curr)
        if self.entropy_flag:
            loss += torch.mean(torch.tensor(self.entropy, dtype=torch.float64).to(self.device).detach())

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        return loss.item()

    def evaluate_advantage(self, memory):
        # We need to call again because critic NN has changed
        rewards, states, _ = self.calc_rewards_states(memory)
        v_curr = self.critic_nn(states).squeeze(-1)
        return rewards - v_curr

    def update_actor_nn(self, advantage, states, actions):
        action_prob = self.policy_nn(states)
        action_prob = action_prob[range(action_prob.shape[0]), actions]
        loss = -torch.mean(torch.log(action_prob) * advantage.detach())
        if self.entropy_flag:
            loss += torch.mean(torch.tensor(self.entropy, dtype=torch.float64).to(self.device).detach())

        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()
        self.entropy = []
        return loss.item()




