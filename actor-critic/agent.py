from agent_control import AgentControl
import numpy as np

class Agent:
    def __init__(self, env, hyperparameters, writer):
        self.agent_control = AgentControl(env, hyperparameters)
        self.writer = writer

        self.ep_reward = 21
        self.total_reward = []
        self.actor_loss = 0
        self.total_actor_loss = []
        self.avg_actor_loss = []
        self.critic_loss = 0
        self.total_critic_loss = []
        self.avg_critic_loss = []

    def choose_action(self, obs):
        return self.agent_control.choose_action(obs)

    def improve_params(self, obs, action, new_obs, reward):
        self.critic_loss = self.agent_control.update_critic_nn(reward, obs, new_obs)
        advantage = self.agent_control.evaluate_advantage(reward, obs, new_obs)
        self.actor_loss = self.agent_control.update_actor_nn(advantage, obs)

        self.ep_reward += reward
        self.total_actor_loss.append(self.actor_loss)
        self.total_critic_loss.append(self.critic_loss)

    def print_state(self, ep_num):
        self.total_reward.append(self.ep_reward)
        self.avg_critic_loss.append(np.mean(self.total_critic_loss))
        self.avg_actor_loss.append(np.mean(self.total_actor_loss))
        print("Episode " + str(ep_num) + " total reward: " + str(self.ep_reward) + " Avg actor loss: " + str(
            np.mean(self.avg_actor_loss[-100:])) + " Avg critic loss: " + str(np.mean(self.avg_critic_loss[-100:])) + " Average reward: " + str(np.mean(self.total_reward[-100:])))
        self.writer.add_scalar('mean_reward', np.mean(self.total_reward[-100:]), ep_num)
        self.writer.add_scalar('ep_reward', self.ep_reward, ep_num)
        self.ep_reward = 21
        self.total_actor_loss = []
        self.total_critic_loss = []
        return np.mean(self.total_reward[-100:])


