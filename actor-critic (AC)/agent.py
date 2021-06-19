from agent_control import AgentControl
import numpy as np
from collections import namedtuple
Memory = namedtuple('Memory', ['obs', 'action', 'new_obs', 'reward'])

class Agent:
    def __init__(self, env, hyperparameters, writer):
        self.agent_control = AgentControl(env, hyperparameters)
        self.n_step = hyperparameters['n-step']
        self.step = 0
        self.memory = []
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

    def improve_params(self, obs, action, new_obs, reward, done):
        # Update episode reward sum
        self.ep_reward += reward
        # If agent didnt reached n-step or end just collect transition and return, do not update NNs
        if (self.step < self.n_step - 1) and not done:
            self.memory.append(Memory(obs=obs, action=action, new_obs=new_obs, reward=reward))
            self.step += 1
            return
        self.memory.append(Memory(obs=obs, action=action, new_obs=new_obs, reward=reward))
        # Calculate discounted rewards, and reverse state|action list because we had to reverse reward list
        rewards, states, actions = self.agent_control.calc_rewards_states(self.memory)
        self.critic_loss = self.agent_control.update_critic_nn(rewards, states)
        # Calculate advantage for every transition
        advantage = self.agent_control.evaluate_advantage(self.memory)
        # Calculate loss and update NN parameters
        self.actor_loss = self.agent_control.update_actor_nn(advantage, states, actions)
        # Reset n-step counter
        self.step = 0
        # Reset memory after each update
        self.memory = []
        # Record losses for statistics
        self.total_actor_loss.append(self.actor_loss)
        self.total_critic_loss.append(self.critic_loss)

    def print_state(self, ep_num):
        self.total_reward.append(self.ep_reward)
        self.avg_critic_loss.append(np.mean(self.total_critic_loss))
        self.avg_actor_loss.append(np.mean(self.total_actor_loss))
        #print("Episode " + str(ep_num) + " total reward: " + str(self.ep_reward) + " Avg actor loss: " + str(
        #    np.mean(self.avg_actor_loss[-100:])) + " Avg critic loss: " + str(np.mean(self.avg_critic_loss[-100:])) + " Average reward: " + str(np.mean(self.total_reward[-100:])))
        if self.writer is not None:
            self.writer.add_scalar('mean_reward', np.mean(self.total_reward[-100:]), ep_num)
        self.ep_reward = 21
        self.total_actor_loss = []
        self.total_critic_loss = []
        self.memory = []
        return np.mean(self.total_reward[-100:])

    def get_policy_nn(self):
        return self.agent_control.get_policy_nn()
