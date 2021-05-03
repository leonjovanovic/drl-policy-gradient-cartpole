"""
1. Initialize the policy parameter θ at random.
2. Generate one trajectory on policy πθ: S1,A1,R2,S2,A2,…,ST.
3. For t=1, 2, … , T:
    Estimate the the return Gt;
4. Update policy parameters: θ←θ+αγtGt∇θlnπθ(At|St)
"""
from agent import Agent
import gym
env = gym.make("CartPole-v1")


HYPERPARAMS = {
    'learning_rate': 5e-5,
    'gamma': 0.99
}

# 1. Initialize the policy parameter θ at random.
agent = Agent(env=env, hyperparameters=HYPERPARAMS)

obs = env.reset()
sum_rewards = 0
action = agent.select_action(obs)

for _ in range(1000):
    env.render()
    action = agent.select_action(obs)
    obs, reward, done, info = env.step(action)
    sum_rewards += reward
    if done:
        observation = env.reset()
        print(sum_rewards)
        sum_rewards = 0
env.close()