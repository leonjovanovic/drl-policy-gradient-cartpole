import torch
import torch.multiprocessing as mp
import gym
from actor_nn import ActorNN
from critic_nn import CriticNN
from test_process import test_process
from train_process import train_process
from agent_control import AgentControl

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
        'num_processes': 1,
        'env_name': "CartPole-v1",
        'max_train_games': 1000,
        'max_test_games': 10,
        'writer_test': False,
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
    actor_optim = torch.optim.Adam(params=shared_model_actor.parameters(), lr=HYPERPARAMETERS['lr_actor'])
    shared_model_critic = CriticNN(env.observation_space.shape[0]).to(device)
    critic_optim = torch.optim.Adam(params=shared_model_critic.parameters(), lr=HYPERPARAMETERS['lr_critic'])
    # Once the tensor/storage is moved to shared_memory, it will be possible to send it to other processes without making any copies.
    # This is a no-op if the underlying storage is already in shared memory and for CUDA tensors. Tensors in shared memory cannot be resized.
    shared_model_actor.share_memory()
    shared_model_critic.share_memory()
    agent_control = AgentControl(HYPERPARAMETERS)
    obs = []
    for rank in range(0, HYPERPARAMETERS['num_processes']):
        obs[rank] = env.reset()
    # List of all workers/processes
    processes = []
    counter_steps = 0
    # We need to create shared counter and lock to safely change value of counter
    end_flag = mp.Value('i', 0)
    memory = mp.Queue()
    ep_num = 0
    for ep_num in range(HYPERPARAMETERS['max_train_games']):
        rewards = []
        states = []
        actions = []
        obs = []
        # We need to start test process which will take current ActorNN params, run 10 episodes and observe rewards, after which params get replaced by next, more updated ActorNN params
        # All train processes stop when test process calculates mean of last 100 episodes to be =>495. After that we run for 90 more episodes to check if last params (used in last 10 episodes)
        # are stable enough to be considered success.
        #p = mp.Process(target=test_process, args=(HYPERPARAMETERS, shared_model_actor, counter, end_flag))
        #p.start()
        #processes.append(p)

        # We will start all training processes passing rank which will determine seed for NN params
        for rank in range(0, HYPERPARAMETERS['num_processes']):
            p = mp.Process(target=train_process, args=(HYPERPARAMETERS, rank, shared_model_actor, memory, obs[rank]))
            p.start()
            processes.append(p)
        # We are waiting for each process to finish collecting Memory
        for p in processes:
            p.join()
            rewards.append(agent_control.get_rewards(memory.get(), shared_model_critic))
            print(rewards)

        #Collect and improve


        # Record number of steps
        counter_steps += HYPERPARAMETERS['num_processes'] * HYPERPARAMETERS['n-step']


# For viewing live progress with tensorboard, open new CMD and type line below:
# tensorboard --logdir "D:\Users\Leon Jovanovic\Documents\Computer Science\Reinforcement Learning\deep-reinforcement-learning-pg-cartpole\A3C\content\runs" --host=127.0.0.1
