import torch.multiprocessing as mp
import gym
from agent import Agent
from test_process import test_process
from torch.utils.tensorboard import SummaryWriter
import time
# -----------------------PARAMETERS---------------------------------
HYPERPARAMETERS = {
    'learning_rate_actor': 0.0007,
    'learning_rate_critic': 0.001,
    'gamma': 0.99,
    'random_seed': -1,
    'entropy': True,
    'entropy_beta': 0.001,
    'n-step': 2,
    'test_counter': 3,
    'env_name': 'CartPole-v1',
    'writer_test': True,
    'writer_train': False,
    'writer_log_dir': 'content/runs/AC-16163232-7,10-n=2-e=001-seed=-1',
    'max_train_games': 1000,
    'max_test_games': 10,
    'print_test_results': True
}
#--------------------------------------------------------------------
if __name__ == '__main__':
    env = gym.make(HYPERPARAMETERS['env_name'])
    obs = env.reset()
    #avg_ep = []
    #writer_train = SummaryWriter(log_dir=HYPERPARAMETERS['writer_log_dir'] + str(time.time())) if HYPERPARAMETERS['writer_train'] else None
    agent = Agent(env=env, hyperparameters=HYPERPARAMETERS, writer=None)

    end_flag = mp.Value('i', 0)
    episode = mp.Value('i', 0)
    #rewards = mp.Array('f', [0]*HYPERPARAMETERS['max_test_games'])
    wait_queue = mp.Queue()
    wait_queue.put(0)

    p = mp.Process(target=test_process, args=(HYPERPARAMETERS, agent.get_policy_nn(), end_flag, wait_queue, episode))
    p.start()

    all_rewards = []
    last_ep = 0

    ep_num = 0
    temp = 0
    start = time.time()
    while ep_num < HYPERPARAMETERS['max_train_games']:
        if ep_num % HYPERPARAMETERS['test_counter'] == 0 and last_ep != ep_num:
            last_ep = ep_num
            wait_queue.put(0)
        # If test process have signalized that we reached neccecary goal (end_flag is shared variable)
        if end_flag.value == 1:
            break
        #env.render()
        action = agent.choose_action(obs)
        new_obs, reward, done, _ = env.step(action)
        if done:
            reward = -20
        agent.improve_params(obs, action, new_obs, reward, done)
        obs = new_obs
        if done:
            obs = env.reset()
            avg_reward = agent.print_state(ep_num)
            ep_num += 1
            episode.value += 1
    p.join()
    env.close()
    # !tensorboard --logdir "D:\Users\Leon Jovanovic\Documents\Computer Science\Reinforcement Learning\deep-reinforcement-learning-pg-cartpole\actor-critic (AC1)\content\runs" --host=127.0.0.1
