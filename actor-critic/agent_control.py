from critic_nn import CriticNN
from policy_nn import PolicyNN

class AgentControl:
    def __init__(self, env, hyperparameters):
        self.input_shape = env.observation_space.shape[0]
        self.output_shape = env.action_space.n
        self.policy_nn = PolicyNN(self.input_shape, self.output_shape)
        self.critic_nn = CriticNN(self.input_shape, self.output_shape)
