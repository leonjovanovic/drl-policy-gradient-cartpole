import gym
import torch
import numpy as np
from collections import namedtuple

Memory = namedtuple('Memory', ['obs', 'action', 'new_obs', 'reward', 'entropy'])

def train_process(hyperparameters, rank, shared_model_actor, memory, continue_queue, end_flag):
    # Create enviroment and agent
    env = gym.make(hyperparameters['env_name'])
    device = 'cpu'# 'cuda' if torch.cuda.is_available() else 'cpu'
    # We set ep_reward to 21 so we can nullify -20 reward we added at the end, so we can compare results as intended
    ep_reward = 21
    obs = env.reset()
    ep_num = 0
    print("Starting process " + str(rank) + str("..."))
    while ep_num < hyperparameters['max_train_games']:
        # If test process have signalized that we reached neccecary goal (end_flag is shared variable)
        if end_flag.value == 1:
            # Process will end if test process alerted train processes that we reached goal
            print("Process " + str(rank) + " ended on episode " + str(ep_num) + "!")
            break
        # Basically simulation of multiple producers and one consumer. Here train process (producer) waits until
        # main (consumer) alerts he is done collecting. There are 'num_processes' continue queues so one process
        # cannot run twice wile other gets blocked
        continue_queue.get()
        # Initially I wanted to create namedtuple Memory with state, reward,... but puting object into memory queue for main
        # to read created Memory leak and main did not release memory until it breaks. Workaround is to separate each field in Memory
        # and to send them separately since type won't be object, but some primitive which won't cause memory leak.
        states = []
        actions = []
        new_states = []
        rewards = []
        entropies = []
        for n_counter in range(hyperparameters['n-step']):
            # Choose action by getting probabilities from ActorNN
            # We send current state as NN input and get two probabilities for each action (in sum of 1)
            action_prob = shared_model_actor(torch.tensor(obs, dtype=torch.double).to(device))
            # We dont take higher probability but take random value of 0 or 1 based on probabilities from NN
            action = np.random.choice(np.array([0, 1]), p=action_prob.cpu().data.numpy())
            # Add sum of probability*log(probability) to list so we can count mean of list when we calculate loss
            # We detach grads since we dont need them when we do loss.backwards() in future
            entropy = 0
            if hyperparameters["entropy_flag"]:
                entropy = -hyperparameters["entropy_coef"] * torch.sum(action_prob * torch.log(action_prob)).detach()
            # Execute chosen action and retrieve new state, reward and if its terminal state
            new_obs, reward, done, _ = env.step(action)
            # To make loss more unrewarding we penalize loss more (instead of default 0)
            if done:
                reward = -20
            ep_reward += reward
            # We append step information to 5 lists which will be sent independently trough memory queue
            states.append(obs)
            actions.append(action)
            new_states.append(new_obs)
            rewards.append(reward)
            entropies.append(entropy)
            # Change new state to be current state so we can continue
            obs = new_obs
            # If we are at the end of episode (terminal state) update and reset counters and state
            if done:
                ep_num += 1
                ep_reward = 21
                obs = env.reset()
                break
        # Each memory queue is unique to each process (see memory_queues[rank] in main.py) so we dont have mix up
        # between different processes. Queue is FIFO and in each iteration we will put 5 times and get 5 times so
        # there won't be any mix up between 2 iteration of same process.
        memory.put(states)
        memory.put(actions)
        memory.put(new_states)
        memory.put(rewards)
        memory.put(entropies)
    env.close()
