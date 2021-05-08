from agent_control import AgentControl
class Agent:
    def __init__(self, env, hyperparameters):
        self.learning_rate = hyperparameters['learning_rate']
        self.gamma = hyperparameters['gamma']
        self.agent_control = AgentControl(env, hyperparameters)
