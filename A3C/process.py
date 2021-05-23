import gym
import torch

from agent import Agent
from torch.utils.tensorboard import SummaryWriter
import time
def train_process(parameters, rank, shared_model_actor, shared_model_critic, counter, lock):
    # Set random seed so each process will get different one
    torch.manual_seed(parameters['seed'] + rank)
    # Create enviroment and agent
    env = gym.make(parameters['env_name'])
    agent = Agent(env, parameters, shared_model_actor, shared_model_critic)
    # If writer flag is True create TensorBoard writer (and start it on independent console to view results)
    writer = SummaryWriter(log_dir=parameters['writer_log_dir'] + str(time.time())) if parameters['writer'] else None

    obs = env.reset()
    ep_num = 0
    while ep_num < parameters['max_worker_games']:
        action = agent.choose_action(obs)
        new_obs, reward, done, _ = env.step(action)
        if done:
            reward = -20
        #------------------------------------------------------------------------------------------------------------------------
        with lock:
            counter.value += 1

        agent.improve(obs, reward, new_obs, action, done)
        obs = new_obs
        if done:
            avg_reward = agent.reset(ep_num, writer, rank)
            ep_num += 1
            obs = env.reset()
            if avg_reward >= 495:
                break
    if writer is not None:
        writer.close()
    env.close()
    # !tensorboard --logdir "D:\Users\Leon Jovanovic\Documents\Computer Science\Reinforcement Learning\deep-reinforcement-learning-pg-cartpole\A3C\content\runs" --host=127.0.0.1

    # Na pocetku koraka prvo prebaciti shared NNs u lokalne NNs, zatim izvrsiti sve i updejtovati shared NN
