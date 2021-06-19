from agent_control import AgentControl
import numpy as np

class Agent:
    def __init__(self, env, hyperparameters, writer):
        self.summary_writer = writer
        self.learning_rate = hyperparameters['learning_rate']
        self.gamma = hyperparameters['gamma']
        self.agent_control = AgentControl(env, hyperparameters)
        self.episode_obs = []
        self.episode_action = []
        self.episode_action_prob = []
        self.episode_new_obs = []
        self.episode_reward = []
        self.total_reward = []
        self.loss = 0
        self.total_loss = []

    def select_action(self, obs):
        return self.agent_control.select_action(obs)

    def add_to_buffer(self, obs, action, new_obs, reward):
        # Add each information to lists
        self.episode_obs.append(obs)
        self.episode_action.append(action)
        self.episode_new_obs.append(new_obs)
        self.episode_reward.append(reward)

    def improve_params(self):
        # We need to first estimate return for each state
        gt = self.estimate_return()
        # Calculate loss based on reward return and update NN parameters
        self.loss = self.agent_control.improve_params(gt, self.episode_obs, self.episode_action)
        # Append to loss list for statistics
        self.total_loss.append(self.loss/len(self.episode_obs))

    def estimate_return(self):
        gt = []
        n = len(self.episode_obs)
        # For each state in episode we calucate return based on actions we took in "future"
        for i in range(n):
            j = i
            power = 0
            temp = 0
            while j < n:
                temp += self.episode_reward[j]*(self.gamma ** power)
                j += 1
                power += 1
            gt.append(temp)
        return gt

    def reset_values(self, ep_num):
        self.episode_reward.append(21)
        self.total_reward.append(sum(self.episode_reward))
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('mean_reward', np.mean(self.total_reward[-100:]), ep_num)
            self.summary_writer.add_scalar('ep_reward', sum(self.episode_reward), ep_num)
        self.episode_obs = []
        self.episode_action = []
        self.episode_action_prob = []
        self.episode_new_obs = []
        self.episode_reward = []
        return np.mean(self.total_reward[-100:])

    def get_policy_nn(self):
        return self.agent_control.get_policy_nn()