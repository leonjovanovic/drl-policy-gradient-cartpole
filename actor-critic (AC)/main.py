import torch.multiprocessing as mp
import gym
from agent import Agent
from test_process import test_process
from torch.utils.tensorboard import SummaryWriter
import time
# -----------------------PARAMETERS---------------------------------
HYPERPARAMETERS = {
    'learning_rate_actor': 0.007,
    'learning_rate_critic': 0.01,
    'gamma': 0.99,
    'random_seed': 12,
    'entropy': True,
    'entropy_beta': 0.001,
    'n-step': 20,
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
    writer_train = SummaryWriter(log_dir=HYPERPARAMETERS['writer_log_dir'] + str(time.time())) if HYPERPARAMETERS['writer_train'] else None
    agent = Agent(env=env, hyperparameters=HYPERPARAMETERS, writer=writer_train)
    # End flag will be controlled by test which will indicate whether train and main process should termiante
    end_flag = mp.Value('i', 0)
    # Episode flag will be shared value of number of episodes done by algorithm
    episode = mp.Value('i', 0)
    # Wait queue forces test process to test NN parameters only when main says so
    wait_queue = mp.Queue()
    wait_queue.put(0)
    # To be sure test process started we create queue which will block main until test is ready
    wait_first_test = mp.Queue()
    # Create and run test process
    p = mp.Process(target=test_process, args=(HYPERPARAMETERS, agent.get_policy_nn(), end_flag, wait_queue, episode, wait_first_test))
    p.start()
    # Remembers last episode we run test process so we dont run test process more than once in same episode
    last_ep = 0
    ep_num = 0
    # Main will be blocked until test is ready
    wait_first_test.get()
    while ep_num < HYPERPARAMETERS['max_train_games']:
        # If we reached 'test_counter' number of episodes and we havent ran test on that episode
        if ep_num % HYPERPARAMETERS['test_counter'] == 0 and last_ep != ep_num:
            last_ep = ep_num
            # Run test process
            wait_queue.put(0)
        # If test process have signalized that we reached neccecary goal (end_flag is shared variable)
        if end_flag.value == 1:
            break
        #env.render()
        # Choose action by getting probabilities from ActorNN
        action = agent.choose_action(obs)
        # Execute chosen action and retrieve new state, reward and if its terminal state
        new_obs, reward, done, _ = env.step(action)
        # To make loss more unrewarding we penalize loss more (instead of default 0)
        if done:
            reward = -20
        # We pass information to agent which will be used to update current parameters of Actor and Critic NN
        agent.improve_params(obs, action, new_obs, reward, done)
        # Change new state to be current state so we can continue
        obs = new_obs
        # If we are at the end of episode (terminal state)
        if done:
            obs = env.reset()
            # Add variables to the list (like episode rewards) and reset those variables
            avg_reward = agent.print_state(ep_num)
            ep_num += 1
            episode.value += 1
    p.join()
    if writer_train is not None:
        writer_train.close()
    env.close()

# For viewing live progress with tensorboard, open new CMD and type line below:
# !tensorboard --logdir "D:\Users\Leon Jovanovic\Documents\Computer Science\Reinforcement Learning\deep-reinforcement-learning-pg-cartpole\actor-critic (AC)\content\runs" --host=127.0.0.1
