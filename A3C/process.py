import gym
from agent import Agent
from torch.utils.tensorboard import SummaryWriter
import time
#----------------------PARAMETERS------------------------------
ENVIROMENT_NAME = "CartPole-v1"
MAX_WORKER_GAMES = 5000
HYPERPARAMETERS = {
    'lr_actor': 0.0001,
    'lr_critic': 0.0005,
    'gamma': 0.9,
    'n-step': 2,
    'entropy_flag': True,
    'entropy_coef': 0.001,
    'seed': 12
}
WRITER_FLAG = True
LOG_DIR = 'content/runs/AC3-16163232-7,10-n=2-e=001-seed=12'
#--------------------------------------------------------------
env = gym.make(ENVIROMENT_NAME)
agent = Agent(env, HYPERPARAMETERS)
writer = SummaryWriter(log_dir=LOG_DIR + str(time.time())) if WRITER_FLAG else None
obs = env.reset()
ep_num = 0
while ep_num < MAX_WORKER_GAMES:
    action = agent.choose_action(obs)
    new_obs, reward, done, _ = env.step(action)
    if done:
        reward = -20
    agent.improve(obs, reward, new_obs, action, done)
    obs = new_obs
    if done:
        avg_reward = agent.reset(ep_num, writer)
        ep_num += 1
        obs = env.reset()
        if avg_reward >= 495:
            break
if writer is not None:
    writer.close()
env.close()
# !tensorboard --logdir "D:\Users\Leon Jovanovic\Documents\Computer Science\Reinforcement Learning\deep-reinforcement-learning-pg-cartpole\A3C\content\runs" --host=127.0.0.1
