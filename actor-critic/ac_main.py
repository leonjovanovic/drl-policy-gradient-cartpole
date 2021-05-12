import gym
from agent import Agent
from torch.utils.tensorboard import SummaryWriter
import time
# -----------------------PARAMETERS---------------------------------
HYPERPARAMETERS = {
    'learning_rate_actor': 0.00001,
    'learning_rate_critic': 0.0001,
    'gamma': 0.99,
    'random_seed': -1,
    'entropy': True
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
while ep_num < MAX_EPISODES:
    #env.render()
    action = agent.choose_action(obs)
    break
    new_obs, reward, done, _ = env.step(action)
    if done:
        reward = -20
    agent.improve_params(obs, action, new_obs, reward)
    obs = new_obs
    if done:
        obs = env.reset()
        agent.print_state(ep_num)
        ep_num += 1
env.close()
writer.close()
# !tensorboard --logdir "D:\Users\Leon Jovanovic\Documents\Computer Science\Reinforcement Learning\deep-reinforcement-learning-pg-cartpole\actor-critic\content\runs" --host=127.0.0.1

# DODATI ENTROPIJU I
#torch.manual_seed(42)
#torch.cuda.manual_seed(42)