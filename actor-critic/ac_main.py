
import gym
import numpy as np

from agent import Agent
from torch.utils.tensorboard import SummaryWriter
import time
# -----------------------PARAMETERS---------------------------------
HYPERPARAMETERS = {
    'learning_rate_actor': 0.0007,
    'learning_rate_critic': 0.0008,
    'gamma': 0.99,
    'random_seed': 12,
    'entropy': True,
    'entropy_beta': 0.001,
    'n-step': 2
}
ENV_NAME = 'CartPole-v1'
WRITER = True
LOG_DIR = 'content/runs/ActorCritic16163232781200012'
MAX_EPISODES = 1000
#--------------------------------------------------------------------
env = gym.make(ENV_NAME)
obs = env.reset()
avg_ep = []
for _ in range(10):
    writer = SummaryWriter(log_dir=LOG_DIR + str(time.time()))
    agent = Agent(env=env, hyperparameters=HYPERPARAMETERS, writer=writer)
    ep_num = 0
    start = time.time()
    while ep_num < MAX_EPISODES:
        #env.render()
        action = agent.choose_action(obs)
        new_obs, reward, done, _ = env.step(action)
        if done:
            reward = -20
        agent.improve_params(obs, action, new_obs, reward, done)
        obs = new_obs
        if done:
            obs = env.reset()
            avg_reward = agent.print_state(ep_num)
            ep_num += 1
            if avg_reward >= 495:
                stop = time.time()
                print("Episodes " + str(ep_num) + " " + str(stop-start))
                avg_ep.append(ep_num)
                break
    writer.close()
    print(np.mean(avg_ep))
env.close()
# !tensorboard --logdir "D:\Users\Leon Jovanovic\Documents\Computer Science\Reinforcement Learning\deep-reinforcement-learning-pg-cartpole\actor-critic\content\runs" --host=127.0.0.1

#1450 seconds