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

        self.device = 'cpu'# 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_nn = PolicyNN(self.input_shape, self.output_shape).to(self.device)
        self.policy_optim = optim.Adam(params=self.policy_nn.parameters(), lr=self.learning_rate_actor)
        self.critic_nn = CriticNN(self.input_shape).to(self.device)
        self.critic_optim = optim.Adam(params=self.critic_nn.parameters(), lr=self.learning_rate_critic)
        self.loss = nn.MSELoss()

    def choose_action(self, obs):
        # We send current state as NN input and get two probabilities for each action (in sum of 1)
        action_prob = self.policy_nn(torch.tensor(obs, dtype=torch.float64).to(self.device))
        # We dont take higher probability but take random value of 0 or 1 based on probabilities from NN
        action = np.random.choice(np.array([0, 1]), p=action_prob.cpu().data.numpy())
        # Add sum of probability*log(probability) to entropy list so we can count mean of entropy list when we calculate loss
        # We detach grads since we dont need them when we do loss.backwards() in future
        if self.entropy_flag:
            self.entropy.append(-self.entropy_beta * torch.sum(action_prob * torch.log(action_prob)).detach())
        return action

    def calc_rewards_states(self, memory):
        # Variable i represents number of rows in memory starting from 0 (number is basically n-step)
        i = len(memory) - 1
        # Calculate Critic value of new state of last step which we will add to accumulated rewards
        v_new = self.critic_nn(torch.tensor(memory[i].new_obs, dtype=torch.float64).to(self.device)).detach()
        rewards = []
        states = []
        actions = []
        # We take just a value of Critic output which will act as base when we add discounted rewards backwards
        temp = v_new.item()
        while i > -1:
            states.append(memory[i].obs)
            actions.append(memory[i].action)
            # For rewards we do backwards discounted sum
            rewards.append(memory[i].reward + self.gamma * temp)
            temp = memory[i].reward + self.gamma * temp
            i -= 1
        # Transform to Tensors so we can use rewards as estimated target
        rewards = torch.tensor(rewards, dtype=torch.float64).to(self.device)
        states = torch.tensor(states, dtype=torch.float64).to(self.device)
        return rewards, states, actions

    def update_critic_nn(self, rewards, states):
        # NN output needs to be squeeze(-1) to lower dimension from matrix to vector of outputs
        v_curr = self.critic_nn(states).squeeze(-1)
        # Calculate MSE loss between target (rewards) and NN output (v_curr)
        loss = self.loss(rewards, v_curr)
        # Add entropy if flag is true
        if self.entropy_flag:
            loss += torch.mean(torch.tensor(self.entropy, dtype=torch.float64).to(self.device).detach())
        # We need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes
        self.critic_optim.zero_grad()
        # Calculate loss derivative
        loss.backward()
        # Update current parameters based on calculated derivatives wtih Adam optimizer
        self.critic_optim.step()
        return loss.item()

    # Estimate advantage as difference between estimated return and actual value
    def evaluate_advantage(self, memory):
        # We need to call again because critic NN has changed
        rewards, states, _ = self.calc_rewards_states(memory)
        v_curr = self.critic_nn(states).squeeze(-1)
        # We estimate advantage as how much Critic NN is right or wrong
        return rewards - v_curr

    # Update Actor NN parameters based on gradient log(action probability) * action probability
    def update_actor_nn(self, advantage, states, actions):
        action_prob = self.policy_nn(states)
        # action_prob is n_step x 2 matrix. We will transfrorm it to n_step x 1 by selecting only probabilities of actions we took
        action_prob = action_prob[range(action_prob.shape[0]), actions]
        # Loss is calculated as log(x) * x for each step. We calculate mean to get single value loss and add minus because torch.log will add additional minus.
        loss = -torch.mean(torch.log(action_prob) * advantage.detach())
        # Add entropy if flag is true
        if self.entropy_flag:
            loss += torch.mean(torch.tensor(self.entropy, dtype=torch.float64).to(self.device).detach())
        # We need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes
        self.policy_optim.zero_grad()
        # Calculate loss derivative
        loss.backward()
        # Update current parameters based on calculated derivatives wtih Adam optimizer
        self.policy_optim.step()
        # We need to reset entropy since we have done one n-step iteration.
        self.entropy = []
        return loss.item()

    def get_policy_nn(self):
        return self.policy_nn
