import torch.optim
import torch.nn as nn
from actor_nn import ActorNN
from critic_nn import CriticNN
import numpy as np

class AgentControl:
    def __init__(self, env, hyperparameters):
        self.gamma = hyperparameters['gamma']
        self.entropy_flag = hyperparameters["entropy_flag"]
        self.entropy_coef = hyperparameters["entropy_coef"]
        self.entropy = []

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.actor_nn = ActorNN(env.observation_space.shape[0], env.action_space.n, hyperparameters["seed"]).to(self.device)
        self.actor_optim = torch.optim.Adam(params=self.actor_nn.parameters(), lr=hyperparameters['lr_actor'])
        self.critic_nn = CriticNN(env.observation_space.shape[0], hyperparameters["seed"]).to(self.device)
        self.critic_optim = torch.optim.Adam(params=self.critic_nn.parameters(), lr=hyperparameters['lr_critic'])
        self.loss = nn.MSELoss()

    def choose_action(self, obs):
        # We send current state as NN input and get two probabilities for each action (in sum of 1)
        action_prob = self.actor_nn(torch.tensor(obs, dtype=torch.double).to(self.device))
        # We dont take higher probability but take random value of 0 or 1 based on probabilities from NN
        action = np.random.choice(np.array([0, 1]), p=action_prob.cpu().data.numpy())
        # Add sum of probability*log(probability) to list so we can count mean of list when we calculate loss
        # We detach grads since we dont need them when we do loss.backwards() in future
        if self.entropy_flag:
            self.entropy.append(-self.entropy_coef * torch.sum(action_prob * torch.log(action_prob)).detach())
        return action

    #Return accumulated discounted estimated reward from memory
    def get_rewards(self, memory):
        # Variable i represents number of rows in memory starting from 0 (number is basically n-step)
        i = len(memory) - 1
        # Calculate Critic value of new state of last step which we will add to accumulated rewards
        v_new = self.critic_nn(torch.tensor(memory[i].new_obs, dtype=torch.float64).to(self.device)).detach()
        rewards = []
        # We take just a value of Critic output which will act as base when we add discounted rewards backwards
        temp = v_new.item()
        while i > -1:
            # For rewards we do backwards discounted sum
            rewards.append(memory[i].reward + self.gamma * temp)
            temp = memory[i].reward + self.gamma * temp
            i -= 1
        # Transform to Tensors so we can use rewards as estimated target
        rewards = torch.tensor(rewards, dtype=torch.float64).to(self.device)
        return rewards

    # Return states and actions in arrays sorted backward. It needs to be backward because rewards have to be calculated from last step.
    # Since we need rewards (target) to match its current state we need to sort states backwards as well.
    def get_states_actions(self, memory):
        # Variable i represents number of rows in memory starting from 0 (number is basically n-step)
        i = len(memory) - 1
        states = []
        actions = []
        while i > -1:
            # For states and actions we create simple lists which we need for critic/actor paralel input
            states.append(memory[i].obs)
            actions.append(memory[i].action)
            i -= 1
        # Transform to Tensors so we can use states as NN input
        states = torch.tensor(states, dtype=torch.float64).to(self.device)
        return states, actions

    # Update Critic NN parameters based on estimated target (rewards) and current value (v_curr)
    def update_critic(self, memory):
        # We call function to get accumulated discounted estimated rewards which will act as target
        # and states which will be used as paralel input for Critic NN
        rewards = self.get_rewards(memory)
        states, _ = self.get_states_actions(memory)
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
    def estimate_advantage(self, memory):
        # We need to call again function because critic NN has changed
        rewards = self.get_rewards(memory)
        states, _ = self.get_states_actions(memory)
        v_curr = self.critic_nn(states).squeeze(-1)
        # We estimate advantage as how much Critic NN is right or wrong
        return (rewards - v_curr).detach()

    # Update Actor NN parameters based on gradient log(action probability) * action probability
    def update_actor(self, memory, advantage):
        # States which will be used as paralel input for Actor NN and actions will be used to know which probability to take
        states, actions = self.get_states_actions(memory)
        action_prob = self.actor_nn(states)
        # action_prob is n_step x 2 matrix. We will transfrorm it to n_step x 1 by selecting only probabilities of actions we took
        action_prob = action_prob[range(action_prob.shape[0]), actions]
        # Loss is calculated as log(x) * x for each step. We calculate mean to get single value loss and add minus because torch.log will add additional minus.
        loss = -torch.mean(torch.log(action_prob) * advantage)

        # Add entropy if flag is true
        if self.entropy_flag:
            loss += torch.mean(torch.tensor(self.entropy, dtype=torch.float64).to(self.device).detach())
        # We need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes
        self.actor_optim.zero_grad()
        # Calculate loss derivative
        loss.backward()
        # Update current parameters based on calculated derivatives wtih Adam optimizer
        self.actor_optim.step()
        # We need to reset entropy since we have done one n-step iteration.
        self.entropy = []
        return loss.item()

