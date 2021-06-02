from agent_nn import PolicyNN
import torch
import torch.nn.functional as F
import numpy as np


class AgentControl:
    def __init__(self, env, hyperparameters):
        self.learning_rate = hyperparameters['learning_rate']
        self.baseline = hyperparameters['baseline']
        self.seed = hyperparameters['random_seed']
        self.device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_nn = PolicyNN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.policy_nn.parameters(), lr=self.learning_rate)

        if self.seed != -1:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)

    def select_action(self, obs):
        action_prob = self.policy_nn(torch.tensor(obs, dtype=torch.double).to(self.device)).cpu()
        action = np.random.choice(np.array([0, 1]), p=action_prob.data.numpy())
        return action

    def improve_params(self, gt, obs, actions):
        gt_tensor = self.subtract_baseline(gt)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        predictions = self.policy_nn(torch.tensor(obs, dtype=torch.double).to(self.device))
        action_prob_tensor = torch.log(predictions)
        action_prob_tensor = action_prob_tensor[range(action_prob_tensor.shape[0]), actions_tensor]
        loss = -torch.sum(action_prob_tensor * gt_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return int(loss.item())

    def subtract_baseline(self, gt):
        gt_tensor = torch.FloatTensor(gt).to(self.device)
        if not self.baseline:
            return gt_tensor
        else:
            return gt_tensor - torch.mean(gt_tensor)

    def get_policy_nn(self):
        return self.policy_nn
