from agent_control import AgentControl
from collections import namedtuple
import numpy as np
Memory = namedtuple('Memory', ['obs', 'action', 'new_obs', 'reward'])
class Agent:
    def __init__(self, env, hyperparameters, shared_model_actor, shared_model_critic):
        self.agent_control = AgentControl(env, hyperparameters, shared_model_actor, shared_model_critic)
        self.n_step = hyperparameters['n-step']
        self.n_counter = 0
        self.memory = []
        self.ep_reward = 21
        self.total_reward = []
        self.critic_loss = 0
        self.actor_loss = 0
        self.total_actor_loss = []
        self.total_critic_loss = []
        self.avg_critic_loss = []
        self.avg_actor_loss = []

    # Choose action based on current state
    def choose_action(self, obs):
        return self.agent_control.choose_action(obs)

    # Improve Actor and Critic neural network parameters
    def improve(self, obs, reward, new_obs, action, done):
        # For each step we add reward
        self.ep_reward += reward
        # For each step we increment n_counter
        self.n_counter += 1
        self.memory.append(Memory(obs, action, new_obs, reward))
        # If n_counter isn't high enough and not last step of episode we just add to memory,
        # if it is we improve params and reset n_counter and memory
        if (self.n_counter < self.n_step) and not done:
            return
        # Update Critic neural network based on memory of past n-step steps
        self.critic_loss = self.agent_control.update_critic(self.memory)
        # Estimate advantage as difference between estimated rewards and current states values from updated Critic NN
        advantage = self.agent_control.estimate_advantage(self.memory)
        # Update Actor neural network based on memory of past n-step steps and advantage
        self.actor_loss = self.agent_control.update_actor(self.memory, advantage)

        # We need to reset n_counter and memory since we have done one n-step iteration.
        self.n_counter = 0
        self.memory = []

        # Add losses to memory
        self.total_actor_loss.append(self.actor_loss)
        self.total_critic_loss.append(self.critic_loss)

    # Print end of episode stats and add it to Tensorboard
    def reset(self, ep_num, writer):
        self.total_reward.append(self.ep_reward)
        self.avg_critic_loss.append(np.mean(self.total_critic_loss))
        self.avg_actor_loss.append(np.mean(self.total_actor_loss))
        if writer is not None:
            writer.add_scalar('mean_reward', np.mean(self.total_reward[-100:]), ep_num)
            writer.add_scalar('ep_reward', self.ep_reward, ep_num)
        self.total_actor_loss = []
        self.total_critic_loss = []
        self.ep_reward = 21
        return np.mean(self.total_reward[-100:])
