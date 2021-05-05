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

    def select_action(self, obs):
        return self.agent_control.select_action(obs)

    def add_to_buffer(self, obs, action, new_obs, reward):
        self.episode_obs.append(obs)
        self.episode_action.append(action)
        self.episode_new_obs.append(new_obs)
        self.episode_reward.append(reward)

    def improve_params(self):
        gt = self.estimate_return()
        self.loss = self.agent_control.improve_params(gt, self.episode_obs, self.episode_action)

    def estimate_return(self):
        gt = []
        n = len(self.episode_obs)
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
        self.total_reward.append(sum(self.episode_reward))
        print("Episode "+str(ep_num)+" total reward: " + str(sum(self.episode_reward)) + " Loss: " + str(self.loss) + " Average reward: " + str(np.mean(self.total_reward[-100:])))
        self.summary_writer.add_scalar('mean_reward', np.mean(self.total_reward[-100:]), ep_num)
        self.episode_obs = []
        self.episode_action = []
        self.episode_action_prob = []
        self.episode_new_obs = []
        self.episode_reward = []

