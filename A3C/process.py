import gym
from agent import Agent
#----------------------PARAMETERS------------------------------
ENVIROMENT_NAME = "CartPole-v1"
MAX_WORKER_GAMES = 1000
HYPERPARAMETERS = {
    'lr_actor': 0.001,
    'lr_critic': 0.001,
    'gamma': 0.9,
    'n-step': 2,
    'entropy_flag': False,
    'entropy_coef': 0.001,
    'seed': 12
}
#--------------------------------------------------------------
env = gym.make(ENVIROMENT_NAME)
agent = Agent(env, HYPERPARAMETERS)
obs = env.reset()
for ep_num in range(MAX_WORKER_GAMES):
    action = agent.choose_action(obs)
    new_obs, reward, done, _ = env.step(action)
    if done:
        reward = -20
    agent.improve(obs, reward, new_obs, action)
    obs = new_obs
    if done:
        agent.reset(ep_num)
        ep_num += 1
        break
        #Proveriti kada ga gasimo

