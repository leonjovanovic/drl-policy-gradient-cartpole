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
        if end_flag.value == 1:
            break
        action = agent.choose_action(obs)
        new_obs, reward, done, _ = env.step(action)
        if done:
            reward = -20
        #------------------------------------------------------------------------------------------------------------------------
        with lock:
            #print(str(counter.value) + " - " + str(rank))
            counter.value += 1

        agent.improve(obs, reward, new_obs, action, done)
        obs = new_obs
        if done:
            avg_reward = agent.reset(ep_num, writer, rank)
            ep_num += 1
            obs = env.reset()
            if avg_reward >= 495:
                break
    print("Ending process " + str(rank) + "!")
    if writer is not None:
        writer.close()
    env.close()

