import gym
from agent import Agent
# -----------------------PARAMETERS---------------------------------
HYPERPARAMETERS = {
    'learning_rate': 0.001,
    'gamma': 0.99,
}
ENV_NAME = 'CartPole-v1'
#--------------------------------------------------------------------
env = gym.make(ENV_NAME)
obs = env.reset()
agent = Agent(env=env, hyperparameters=HYPERPARAMETERS)
ep_num = 0
for _ in range(1100):
    env.render()
    action = agent.choose_action(obs)
    new_obs, reward, done, _ = env.step(action)
    agent.improve_params(obs, action, new_obs, reward)
    obs = new_obs
    if done:
        obs = env.reset()
        agent.reset_values(ep_num)
        ep_num += 1
env.close()
