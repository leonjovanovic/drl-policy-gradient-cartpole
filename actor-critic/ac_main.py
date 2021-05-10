import gym
from agent import Agent
# -----------------------PARAMETERS---------------------------------
HYPERPARAMETERS = {
    'learning_rate_actor': 0.001,
    'learning_rate_critic': 0.01,
    'gamma': 0.9,
}
ENV_NAME = 'CartPole-v1'
#--------------------------------------------------------------------
env = gym.make(ENV_NAME)
obs = env.reset()
agent = Agent(env=env, hyperparameters=HYPERPARAMETERS)
ep_num = 0
while ep_num < 5000:
    #env.render()
    action = agent.choose_action(obs)
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
