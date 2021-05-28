import torch.multiprocessing as mp
import gym
import numpy as np
from agent import Agent
from test_process import test_process
from torch.utils.tensorboard import SummaryWriter
import time
# -----------------------PARAMETERS---------------------------------
HYPERPARAMETERS = {
    'learning_rate_actor': 0.0007,
    'learning_rate_critic': 0.001,
    'gamma': 0.99,
    'random_seed': 12,
    'entropy': True,
    'entropy_beta': 0.001,
    'n-step': 2,
    'test_counter': 10,
    'env_name': 'CartPole-v1',
    'writer_test': True,
    'writer_train': False,
    'writer_log_dir': 'content/runs/AC1-16163232-7,10-n=2-e=001-seed=12',
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
    writer_test = SummaryWriter(log_dir=HYPERPARAMETERS['writer_log_dir'] + str(time.time())) if HYPERPARAMETERS['writer_test'] else None
    agent = Agent(env=env, hyperparameters=HYPERPARAMETERS, writer=None)

    counter = mp.Value('i', 0)
    end_flag = mp.Value('i', 0)
    rewards = mp.Array('f', [0]*HYPERPARAMETERS['max_test_games'])
    all_rewards = []
    last_ep = 0

    ep_num = 0
    start = time.time()
    while ep_num < HYPERPARAMETERS['max_train_games']:
        if ep_num % HYPERPARAMETERS['test_counter'] == 0 and last_ep != ep_num:
            last_ep = ep_num
            p = mp.Process(target=test_process, args=(HYPERPARAMETERS, agent.get_policy_nn(), counter, end_flag, rewards))
            p.start()
            p.join()
            all_rewards.append(rewards)
            if writer_test is not None:
                writer_test.add_scalar('mean_reward', np.mean(all_rewards[-100:]), counter.value)
        # If test process have signalized that we reached neccecary goal (end_flag is shared variable)
        if end_flag.value == 1:
            break
        #env.render()
        action = agent.choose_action(obs)
        new_obs, reward, done, _ = env.step(action)
        if done:
            reward = -20

        counter.value += 1

        agent.improve_params(obs, action, new_obs, reward, done)
        obs = new_obs
        if done:
            obs = env.reset()
            avg_reward = agent.print_state(ep_num)
            ep_num += 1
            #if avg_reward >= 495:
            #    stop = time.time()
            #    print("Episodes " + str(ep_num) + " " + str(stop-start))
            #    avg_ep.append(ep_num)
            #    break

    if writer_test is not None:
        writer_test.close()
    env.close()
    # !tensorboard --logdir "D:\Users\Leon Jovanovic\Documents\Computer Science\Reinforcement Learning\deep-reinforcement-learning-pg-cartpole\actor-critic (AC1)\content\runs" --host=127.0.0.1
