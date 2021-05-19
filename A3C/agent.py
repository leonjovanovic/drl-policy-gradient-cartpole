from agent_control import AgentControl
from collections import namedtuple
Memory = namedtuple('Memory', ['obs', 'action', 'new_obs', 'reward'])
class Agent:
    def __init__(self, env, hyperparameters):
        self.agent_control = AgentControl(env, hyperparameters)
        self.n_step = hyperparameters['n-step']
        self.n_counter = 0
        self.memory = []

    # Choose action based on current state
    def choose_action(self, obs):
        return self.agent_control.choose_action(obs)

    # Improve Actor and Critic neural network parameters
    def improve(self, obs, reward, new_obs, action):
        # For each step we increment n_counter
        self.n_counter += 1
        self.memory.append(Memory(obs, action, new_obs, reward))
        # If n_counter isn't high enough we just add to memory, if it is we improve params
        # and reset n_counter and memory
        if self.n_counter < self.n_step:
            return
        # Update Critic neural network based on memory of past n-step steps
        self.agent_control.update_critic(self.memory)
        # Estimate advantage as difference between estimated rewards and current states values from updated Critic NN
        advantage = self.agent_control.estimate_advantage(self.memory)
        # Update Actor neural network based on memory of past n-step steps and advantage
        self.agent_control.update_actor(self.memory, advantage)

        # We need to reset n_counter and memory since we have done one n-step iteration.
        self.n_counter = 0
        self.memory = []

    # Print end of episode stats and add it to Tensorboard
    def reset(self, ep_num):
        print("End")
