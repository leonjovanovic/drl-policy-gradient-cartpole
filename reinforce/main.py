from test_process import test_process
from agent import Agent
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import gym
import time

#-----------------------------PARAMETERS-----------------------------
HYPERPARAMETERS = {
    'learning_rate': 0.003,
    'gamma': 0.99,
    'random_seed': 12,
    'baseline': True,
    'test_counter': 8,
    'env_name': 'CartPole-v1',
    'writer_test': True,
    'writer_train': False,
    'writer_log_dir': 'content/runs/REINFORCE-3232-3-baseline-seed=-1',
    'max_train_games': 5000,
    'max_test_games': 10,
    'print_test_results': True
}
#--------------------------------------------------------------------
if __name__ == '__main__':
    # Create TensorBoard writer that will create graphs
    writer_train = SummaryWriter(log_dir=HYPERPARAMETERS['writer_log_dir'] + str(time.time())) if HYPERPARAMETERS['writer_train'] else None
    # Create enviroment
    env = gym.make(HYPERPARAMETERS['env_name'])
    # Initialize the policy parameter θ at random.
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
    obs = env.reset()
    ep_num = 0
    # Main will be blocked until test is ready
    wait_first_test.get()
    while ep_num < HYPERPARAMETERS['max_train_games']:
        # If test process have signalized that we reached neccecary goal (end_flag is shared variable)
        if end_flag.value == 1:
            break
        env.render()
        # Give current state to NN and get action from it
        action = agent.select_action(obs)
        # Take that action and retreive next state, reward and is it terminal state
        new_obs, reward, done, _ = env.step(action)
        if done:
            reward = -20
        # Until we reach end of episode, store transitions
        agent.add_to_buffer(obs, action, new_obs, reward)
        obs = new_obs
        if done:
            # For each step in episode we need to estimate return Gt and update policy parameters
            agent.improve_params()
            avg_reward = agent.reset_values(ep_num)
            obs = env.reset()
            ep_num += 1
            episode.value += 1
            # Let test process test new parameters
            wait_queue.put(0)
    # Wait for test process to end before terminating main
    p.join()
    if writer_train is not None:
        writer_train.close()
    env.close()
    # !tensorboard --logdir "D:\Users\Leon Jovanovic\Documents\Computer Science\Reinforcement Learning\deep-reinforcement-learning-pg-cartpole\reinforce\content\runs" --host=127.0.0.1
    # !tensorboard --inspect --logdir "D:\Users\Leon Jovanovic\Documents\Computer Science\Reinforcement Learning\deep-reinforcement-learning-pg-cartpole\reinforce\content\runs"
