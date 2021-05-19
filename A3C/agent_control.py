
class AgentControl:
    def __init__(self, env, hyperparameters):
        self.lr_actor = hyperparameters['lr_actor']
        self.lr_critic = hyperparameters['lr_critic']
        self.gamma = hyperparameters['gamma']
        self.entropy_flag = hyperparameters["entropy_flag"]
        self.entropy = hyperparameters["entropy"]

        #self.nn