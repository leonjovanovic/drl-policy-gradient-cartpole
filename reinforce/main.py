from agent import Agent
from torch.utils.tensorboard import SummaryWriter
import gym
import time

#-----------------------------PARAMETERS-----------------------------
HYPERPARAMS = {
    'learning_rate': 0.001,
    'gamma': 0.99,
    'baseline': True
}
MAX_EPISODES = 5000
LOG_DIR = 'content/runs'
NAME = "VanillaPG"
ENV_NAME = "CartPole-v1"
#--------------------------------------------------------------------
# Create TensorBoard writer that will create graphs
writer = SummaryWriter(log_dir=LOG_DIR + '/' + NAME + str(time.time()))
# Create enviroment
env = gym.make(ENV_NAME)
# Initialize the policy parameter θ at random.
agent = Agent(env=env, hyperparameters=HYPERPARAMS, writer=writer)

obs = env.reset()
ep_num = 0
start = time.time()
while ep_num < MAX_EPISODES:
    #env.render()
    # Give current state to NN and get action from it
    action = agent.select_action(obs)
    # Take that action and retreive next state, reward and is it terminal state
    new_obs, reward, done, _ = env.step(action)
    if done:
        reward = -20
    # Until we reach end of episode, store transitions
    agent.add_to_buffer(obs, action, new_obs, reward)
    obs = new_obs
    if done:
        # For each step in episode we need to estimate return Gt and update policy parameters
        agent.improve_params()
        avg_reward = agent.reset_values(ep_num)
        obs = env.reset()
        ep_num += 1
        if avg_reward >= 495:
            stop = time.time()
            print(stop-start)
            break

env.close()
writer.close()
# !tensorboard --logdir "D:\Users\Leon Jovanovic\Documents\Computer Science\Reinforcement Learning\deep-reinforcement-learning-pg-cartpole\reinforce\content\runs" --host=127.0.0.1
# !tensorboard --inspect --logdir "D:\Users\Leon Jovanovic\Documents\Computer Science\Reinforcement Learning\deep-reinforcement-learning-pg-cartpole\reinforce\content\runs"
