from agent_control import AgentControl
import numpy as np

class Agent:
    def __init__(self, env, hyperparameters):
        self.agent_control = AgentControl(env, hyperparameters)

        self.ep_reward = 0
        self.total_reward = []
        self.actor_loss = 0
        self.critic_loss = 0

    def choose_action(self, obs):
        return self.agent_control.choose_action(obs)

    def improve_params(self, obs, action, new_obs, reward):
        self.critic_loss = self.agent_control.update_critic_nn(reward, obs, new_obs)
        advantage = self.agent_control.evaluate_advantage(reward, obs, new_obs)
        self.actor_loss = self.agent_control.update_actor_nn(advantage)

        self.ep_reward += reward

    def print_state(self, ep_num):
        self.total_reward.append(self.ep_reward)
        print("Episode " + str(ep_num) + " total reward: " + str(self.ep_reward) + " Actor loss: " + str(
            self.actor_loss) + " Critic loss: " + str(self.critic_loss) + " Average reward: " + str(np.mean(self.total_reward[-100:])))
        self.ep_reward = 0


