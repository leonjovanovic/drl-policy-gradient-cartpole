import gym
import torch

from agent import Agent
from torch.utils.tensorboard import SummaryWriter
import time
def train_process(parameters, rank, shared_model_actor, shared_model_critic, counter, lock, end_flag):
    # Set random seed so each process will get different one
    torch.manual_seed(parameters['seed'] + rank)
    # Create enviroment and agent
    env = gym.make(parameters['env_name'])
    agent = Agent(env, parameters, shared_model_actor, shared_model_critic)
    # If writer flag is True create TensorBoard writer (and start it on independent console to view results)
    writer = SummaryWriter(log_dir=parameters['writer_log_dir'] + str(time.time())) if parameters['writer_train'] else None

    obs = env.reset()
    ep_num = 0
    print("Starting process " + str(rank) + str("..."))
    while ep_num < parameters['max_train_games']:
        # If test process have signalized that we reached neccecary goal (end_flag is shared variable)
        if end_flag.value == 1:
            break
        # Choose action by getting probabilities from ActorNN
        action = agent.choose_action(obs)
        # Execute chosen action and retrieve new state, reward and if its terminal state
        new_obs, reward, done, _ = env.step(action)
        # To make loss more unrewarding we penalize loss more (instead of default 0)
        if done:
            reward = -20
        # If we want to change value of shared variable we need to get lock so no one can change value until this process finishes changing value
        with lock:
            counter.value += 1
        # We pass information to agent which will be used to update current parameters of Actor and Critic NN
        agent.improve(obs, reward, new_obs, action, done)
        # Change new state to be current state so we can continue
        obs = new_obs
        # If we are at the end of episode (terminal state)
        if done:
            # Add variables to the list (like episode rewards) and reset those variables
            agent.reset(ep_num, writer)
            ep_num += 1
            obs = env.reset()
    # Process will end if test process alerted train processes that we reached goal
    print("Process " + str(rank) + " ended with episode " + str(ep_num) + "!")
    # Close writer and enviroments at the end
    if writer is not None:
        writer.close()
    env.close()
