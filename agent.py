from agent_control import AgentControl

class Agent:
    def __init__(self, env, hyperparameters):
        self.learning_rate = hyperparameters['learning_rate']
        self.agent_control = AgentControl(env, hyperparameters)

    def select_action(self, obs):
        print(1)
        return self.agent_control.select_action(obs)
