import torch
import torch.multiprocessing as mp
import gym
from actor_nn import ActorNN
from critic_nn import CriticNN
from test_process import test_process
from train_process import train_process

if __name__ == '__main__':
    #----------------------PARAMETERS-------------------------------
    MAX_WORKER_GAMES = 1000
    HYPERPARAMETERS = {
        'lr_actor': 0.0002,
        'lr_critic': 0.0003,
        'gamma': 0.99,
        'n-step': 2,
        'entropy_flag': True,
        'entropy_coef': 0.001,
        'seed': 12,
        'num_processes': 11,
        'env_name': "CartPole-v1",
        'max_train_games': 1000,
        'max_test_games': 10,
        'writer_test': True,
        'writer_train': False,
        'writer_log_dir': 'content/runs/AC3-16163232-2,3-n=2-e=001-seed=12++',
        'print_test_results': True
    }
    #---------------------------------------------------------------
    # Set manuel seed so other processes dont get same
    torch.manual_seed(HYPERPARAMETERS['seed'])
    # Create enviroment so we can get state (input) size and action space (output) size
    env = gym.make(HYPERPARAMETERS['env_name'])
    # We need to create two models which will be shared across workers
    device = 'cpu'# 'cuda' if torch.cuda.is_available() else 'cpu'
    shared_model_actor = ActorNN(env.observation_space.shape[0], env.action_space.n).to(device)
    shared_model_critic = CriticNN(env.observation_space.shape[0]).to(device)
    # Once the tensor/storage is moved to shared_memory, it will be possible to send it to other processes without making any copies.
    # This is a no-op if the underlying storage is already in shared memory and for CUDA tensors. Tensors in shared memory cannot be resized.
    shared_model_actor.share_memory()
    shared_model_critic.share_memory()

    # List of all workers/processes
    processes = []
    # We need to create shared counter and lock to safely change value of counter
    counter = mp.Value('i', 0)
    end_flag = mp.Value('i', 0)
    lock = mp.Lock()

    # We need to start test process which will take current ActorNN params, run 10 episodes and observe rewards, after which params get replaced by next, more updated ActorNN params
    # All train processes stop when test process calculates mean of last 100 episodes to be =>495. After that we run for 90 more episodes to check if last params (used in last 10 episodes)
    # are stable enough to be considered success.
    p = mp.Process(target=test_process, args=(HYPERPARAMETERS, shared_model_actor, counter, end_flag))
    p.start()
    processes.append(p)

    # We will start all training processes passing rank which will determine seed for NN params
    for rank in range(0, HYPERPARAMETERS['num_processes']):
        p = mp.Process(target=train_process, args=(HYPERPARAMETERS, rank, shared_model_actor, shared_model_critic, counter, lock, end_flag))
        p.start()
        processes.append(p)
    # We are waiting for each process to finish
    for p in processes:
        p.join()

# For viewing live progress with tensorboard, open new CMD and type line below:
# tensorboard --logdir "D:\Users\Leon Jovanovic\Documents\Computer Science\Reinforcement Learning\deep-reinforcement-learning-pg-cartpole\A3C\content\runs" --host=127.0.0.1
