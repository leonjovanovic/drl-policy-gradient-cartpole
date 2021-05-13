import gym
from agent import Agent
from torch.utils.tensorboard import SummaryWriter
import time
# -----------------------PARAMETERS---------------------------------
HYPERPARAMETERS = {
    'learning_rate_actor': 0.0001,
    'learning_rate_critic': 0.0002,
    'gamma': 0.99,
    'random_seed': -1,
    'entropy': True,
    'entropy_beta': 0.1
}
ENV_NAME = 'CartPole-v1'
WRITER = True
LOG_DIR = 'content/runs/ActorCritic'
MAX_EPISODES = 4000
#--------------------------------------------------------------------
env = gym.make(ENV_NAME)
obs = env.reset()
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
    agent.improve_params(obs, action, new_obs, reward)
    obs = new_obs
    if done:
        obs = env.reset()
        avg_reward = agent.print_state(ep_num)
        ep_num += 1
        if avg_reward >= 495:
            stop = time.time()
            print(stop-start)
            break
env.close()
writer.close()
# !tensorboard --logdir "D:\Users\Leon Jovanovic\Documents\Computer Science\Reinforcement Learning\deep-reinforcement-learning-pg-cartpole\actor-critic\content\runs" --host=127.0.0.1

#1450 seconds