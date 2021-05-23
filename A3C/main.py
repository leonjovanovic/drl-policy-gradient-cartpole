import torch
import torch.multiprocessing as mp
import gym
from actor_nn import ActorNN
from critic_nn import CriticNN
from process import train_process

if __name__ == '__main__':
    #----------------------PARAMETERS------------------------------
    MAX_WORKER_GAMES = 1000
    HYPERPARAMETERS = {
        'lr_actor': 0.0002,
        'lr_critic': 0.0003,
        'gamma': 0.99,
        'n-step': 5,
        'entropy_flag': True,
        'entropy_coef': 0.001,
        'seed': 12,
        'num_processes': 12,
        'env_name': "CartPole-v1",
        'max_worker_games': 1000,
        'writer': True,
        'writer_log_dir': 'content/runs/AC3-16163232-2,3-n=4-e=001-seed=12++'
    }
    #--------------------------------------------------------------
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
    # We need to create Value and Lock for classic Semaphore
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    #p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
    #p.start()
    #processes.append(p)

    # We will start all processes passing rank which will determine seed for NN params
    for rank in range(0, HYPERPARAMETERS['num_processes']):
        p = mp.Process(target=train_process, args=(HYPERPARAMETERS, rank, shared_model_actor, shared_model_critic, counter, lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
