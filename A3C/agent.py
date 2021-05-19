from agent_control import AgentControl
class Agent:
    def __init__(self, env, hyperparameters):
        self.agent_control = AgentControl(env, hyperparameters)
