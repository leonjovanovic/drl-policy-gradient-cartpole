import torch.optim
import torch.nn as nn

class AgentControl:
    def __init__(self, hyperparameters):
        self.gamma = hyperparameters['gamma']

        self.device = 'cpu'# 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss = nn.MSELoss()

    #Return accumulated discounted estimated reward from memory
    def get_rewards(self, old_rewards, new_states, critic_nn):
        # Variable i represents number of rows in memory starting from 0 (number is basically n-step)
        i = len(old_rewards) - 1
        # Calculate Critic value of new state of last step which we will add to accumulated rewards
        v_new = critic_nn(torch.tensor(new_states[i], dtype=torch.float64).to(self.device)).detach()
        rewards = []
        # We take just a value of Critic output which will act as base when we add discounted rewards backwards
        temp = v_new.item()
        while i > -1:
            # For rewards we do backwards discounted sum
            rewards.append(old_rewards[i] + self.gamma * temp)
            temp = old_rewards[i] + self.gamma * temp
            i -= 1
        # Transform to Tensors so we can use rewards as estimated target
        #rewards = torch.tensor(rewards, dtype=torch.float64).to(self.device)
        return rewards

    # Return states and actions in arrays sorted backward. It needs to be backward because rewards have to be calculated from last step.
    # Since we need rewards (target) to match its current state we need to sort states backwards as well.
    def get_states_actions_entropies(self, st, ac, en):
        # Variable i represents number of rows in memory starting from 0 (number is basically n-step)
        i = len(st) - 1
        states = []
        actions = []
        entropies = []
        while i > -1:
            # For states and actions we create simple lists which we need for critic/actor paralel input
            states.append(st[i])
            actions.append(ac[i])
            entropies.append(en[i])
            i -= 1
        return states, actions, entropies

    # Update Critic NN parameters based on estimated target (rewards) and current value (v_curr)
    def update_critic(self, rewards, states, entropies, critic_nn, critic_optim):
        rewards = torch.tensor(rewards, dtype=torch.float64).to(self.device)
        states = torch.tensor(states, dtype=torch.float64).to(self.device)
        # NN output needs to be squeeze(-1) to lower dimension from matrix to vector of outputs
        v_curr = critic_nn(states).squeeze(-1)
        # Calculate MSE loss between target (rewards) and NN output (v_curr)
        loss = self.loss(rewards, v_curr)
        # Add entropy, if flag is False it will add 0
        loss += torch.mean(torch.tensor(entropies, dtype=torch.float64).to(self.device).detach())
        # We need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes
        critic_optim.zero_grad()
        # Calculate loss derivative
        loss.backward()
        # Update current parameters based on calculated derivatives wtih Adam optimizer
        critic_optim.step()
        return loss.item()

    # Estimate advantage as difference between estimated return and actual value
    def estimate_advantage(self, rewards, states, critic_nn):
        rewards = torch.tensor(rewards, dtype=torch.float64).to(self.device)
        states = torch.tensor(states, dtype=torch.float64).to(self.device)
        v_curr = critic_nn(states).squeeze(-1)
        # We estimate advantage as how much Critic NN is right or wrong
        return (rewards - v_curr).detach()

    # Update Actor NN parameters based on gradient log(action probability) * action probability
    def update_actor(self, states, actions, entropies, advantage, actor_nn, actor_optim):
        states = torch.tensor(states, dtype=torch.float64).to(self.device)
        action_prob = actor_nn(states)
        # action_prob is n_step x 2 matrix. We will transfrorm it to n_step x 1 by selecting only probabilities of actions we took
        action_prob = action_prob[range(action_prob.shape[0]), actions]
        # Loss is calculated as log(x) * x for each step. We calculate mean to get single value loss and add minus because torch.log will add additional minus.
        loss = -torch.mean(torch.log(action_prob) * advantage)
        # Add entropy, if flag is False it will add 0
        loss += torch.mean(torch.tensor(entropies, dtype=torch.float64).to(self.device).detach())
        # We need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes
        actor_optim.zero_grad()
        # Calculate loss derivative
        loss.backward()
        # Since we calculated all grads on local actor nn we do not have grads on shared actor nn
        # So we need to copy grads for each NN parameter from local to global actor nn
        #ensure_shared_grads(self.actor_nn, self.shared_actor_nn)
        # Update current parameters based on calculated derivatives wtih Adam optimizer
        actor_optim.step()
        # We need to reset entropy since we have done one n-step iteration.
        return loss.item()
