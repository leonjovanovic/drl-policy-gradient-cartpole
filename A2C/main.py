import torch
import torch.multiprocessing as mp
import gym
from actor_nn import ActorNN
from critic_nn import CriticNN
from test_process import test_process
from train_process import train_process
from agent_control import AgentControl
import gc

if __name__ == '__main__':
    #----------------------PARAMETERS-------------------------------
    MAX_WORKER_GAMES = 1000
    HYPERPARAMETERS = {
        'lr_actor': 0.0006,
        'lr_critic': 0.0007,
        'gamma': 0.99,
        'n-step': 7,
        'entropy_flag': True,
        'entropy_coef': 0.001,
        'seed': 12,
        'num_processes': 10,
        'env_name': "CartPole-v1",
        'max_train_games': 1000,
        'max_test_games': 10,
        'writer_test': False,
        'writer_train': False,
        'writer_log_dir': 'content/runs/A2C-16163232-6,7-n=6-e=001-seed=12-proc=10',
        'print_test_results': True
    }
    #---------------------------------------------------------------
    # Set manuel seed so other processes dont get same
    torch.manual_seed(HYPERPARAMETERS['seed'])
    # Create enviroment so we can get state (input) size and action space (output) size
    env = gym.make(HYPERPARAMETERS['env_name'])
    # We need to create two models which will be shared across workers and two Adam optimizers to update parameters
    device = 'cpu'# 'cuda' if torch.cuda.is_available() else 'cpu'
    shared_model_actor = ActorNN(env.observation_space.shape[0], env.action_space.n).to(device)
    shared_model_critic = CriticNN(env.observation_space.shape[0]).to(device)
    actor_optim = torch.optim.Adam(params=shared_model_actor.parameters(), lr=HYPERPARAMETERS['lr_actor'])
    critic_optim = torch.optim.Adam(params=shared_model_critic.parameters(), lr=HYPERPARAMETERS['lr_critic'])
    # Once the tensor/storage is moved to shared_memory, it will be possible to send it to other processes without making any copies.
    # This is a no-op if the underlying storage is already in shared memory and for CUDA tensors. Tensors in shared memory cannot be resized.
    shared_model_actor.share_memory()
    shared_model_critic.share_memory()
    # We create AgentControl so we can use its functions to update NNs and calculate Advantage
    agent_control = AgentControl(HYPERPARAMETERS)
    # List of all workers/processes
    processes = []
    # We need to create 'num_processes' different queues for sending memory to make sure each process sent one iteration
    memory_queues = []
    for rank in range(0, HYPERPARAMETERS['num_processes']):
        memory_queues.append(mp.Queue())
    # We need to create 'num_processes' different queues for allowing train process to continue to make sure each process get pass only once
    continue_queues = []
    for rank in range(0, HYPERPARAMETERS['num_processes']):
        continue_queues.append(mp.Queue())
        # We need to allow them to pass initially once
        continue_queues[rank].put(rank)
    # This queue is made so test process isn't testing NNs non-stop, but only when we tell it to do so
    wait_test = mp.Queue()
    wait_test.put(1)
    # This shared value is set only by test process when it reaches goal, rest of the processes are terminating if this value is 1
    end_flag = mp.Value('i', 0)
    # Shared value for passing number of steps to test process
    counter_steps = mp.Value('i', 0)
    # We need to start test process which will take current ActorNN params, run 10 episodes and observe rewards, after which params get replaced by next, more updated ActorNN params
    # All train processes stop when test process calculates mean of last 100 episodes to be =>495. After that we run for 90 more episodes to check if last params (used in last 10 episodes)
    # are stable enough to be considered success.
    p = mp.Process(target=test_process, args=(HYPERPARAMETERS, shared_model_actor, counter_steps, end_flag, wait_test))
    p.start()
    processes.append(p)
    # We will start all training processes passing rank which will determine number of each process
    for rank in range(0, HYPERPARAMETERS['num_processes']):
        p = mp.Process(target=train_process, args=(HYPERPARAMETERS, rank, shared_model_actor, memory_queues[rank], continue_queues[rank], end_flag))
        p.start()
        processes.append(p)
    # We need counter so we will call test process after 20 iterations
    counter = -1
    while True:
        counter += 1
        all_rewards = []
        all_states = []
        all_actions = []
        all_entropies = []
        # We are waiting for each process to finish collecting data
        for rank in range(0, HYPERPARAMETERS['num_processes']):
            # We call garbage collector because of previous Memory leak problem. We should remove it and test it since problem should be fixed
            gc.collect()
            # If test process have signalized that we reached neccecary goal we need to terminate
            if end_flag.value == 1:
                break
            # Collect states, actions, new_states, rewards and entropies. Queue.get() is blocking and it will wait
            # until process[rank] put each data in the queue[rank]
            states = memory_queues[rank].get()
            actions = memory_queues[rank].get()
            new_states = memory_queues[rank].get()
            rewards = memory_queues[rank].get()
            entropies = memory_queues[rank].get()
            # We accumulate discounted estimated reward
            all_rewards.extend(agent_control.get_rewards(rewards, new_states, shared_model_critic))
            # We need to flip states, action and entropies list backwards because of way we calculate discounted reward (look agent_control.get_rewards())
            st, ac, en = agent_control.get_states_actions_entropies(states, actions, entropies)
            all_states.extend(st)
            all_actions.extend(ac)
            all_entropies.extend(en)
        # Update Critic NN parameters
        critic_loss = agent_control.update_critic(all_rewards, all_states, all_entropies, shared_model_critic, critic_optim)
        #CHECK IF WE NEED TO CALCULATE REWARDS AGAIN BECAUSE CRITIC CHANGED IN LAST LINE
        # In previous algorithms (AC, A3C) we calculated Advantage on rewards we recalculated on new Critic NN parameters.
        # Here it is possible but very inefficient because of way we collect data, so we will calculate advantage based
        # on original Critic NN (before last line where we update it)
        # Disclamer: Actor-Critic algorithm varies in a way that in some pseudocodes or lectures advantage is calculated with old and sometimes with new Critic NN parameters.
        advantage = agent_control.estimate_advantage(all_rewards, all_states, shared_model_critic)
        # Update Actor NN parameters
        actor_loss = agent_control.update_actor(all_states, all_actions, all_entropies, advantage, shared_model_actor, actor_optim)
        # Record number of steps
        counter_steps.value += HYPERPARAMETERS['num_processes'] * HYPERPARAMETERS['n-step']
        # After updating NNs we are telling each process to continue by adding something to their continue queue so they get unblocked
        for rank in range(0, HYPERPARAMETERS['num_processes']):
            continue_queues[rank].put(rank)
        # If its 20th iteration call test process to test current Actor NN parameters
        if counter == 20:
            counter = 0
            wait_test.put(1)
        # If test process have signalized that we reached neccecary goal (end_flag is shared variable)
        if end_flag.value == 1:
            break
    # Test alerted that we reached our goal and we broke out of while loop. Now we wait for all train and test processes to end before main terminates
    for p in processes:
        p.join()
# For viewing live progress with tensorboard, open new CMD and type line below:
# tensorboard --logdir "D:\Users\Leon Jovanovic\Documents\Computer Science\Reinforcement Learning\deep-reinforcement-learning-pg-cartpole\A2C\content\runs" --host=127.0.0.1
