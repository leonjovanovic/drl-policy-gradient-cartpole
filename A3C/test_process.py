import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
from actor_nn import ActorNN


def test_process(parameters, shared_model_actor, counter, end_flag):
    # Create enviroment and agent
    env = gym.make(parameters['env_name'])
    # If writer flag is True create TensorBoard writer (and start it on independent console to view results)
    writer = SummaryWriter(log_dir=parameters['writer_log_dir'] + str(time.time())) if parameters['writer_test'] else None

    device = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ActorNN(env.observation_space.shape[0], env.action_space.n).to(device)

    obs = env.reset()
    ep_reward = 21
    all_rewards = []
    while True:
        #print("Testing...")
        ep_num = 0
        model.load_state_dict(shared_model_actor.state_dict())
        while ep_num < parameters['max_test_games']:
            # We send current state as NN input and get two probabilities for each action (in sum of 1)
            action_prob = model(torch.tensor(obs, dtype=torch.double).to(device))
            # We dont take higher probability but take random value of 0 or 1 based on probabilities from NN
            action = np.random.choice(np.array([0, 1]), p=action_prob.cpu().data.numpy())
            # We take action we got from global NN and take new state, reward and is it terminal state
            new_obs, reward, done, _ = env.step(action)
            if done:
                reward = -20
            ep_reward += reward
            obs = new_obs
            if done:
                all_rewards.append(ep_reward)
                #print("Test process " + str(counter.value) + " Episode " + str(ep_num) + " total reward: " + str(ep_reward) + " Average reward: " + str(np.mean(all_rewards[-100:])))
                obs = env.reset()
                ep_num += 1
                ep_reward = 21

        if writer is not None:
            writer.add_scalar('mean_reward', np.mean(all_rewards[-100:]), counter.value)
        if np.mean(all_rewards[-100:]) >= 495:
            break

    end_flag.value = 1
    print("End testing!--------------------------------------------")

    while ep_num < 100:
        # We send current state as NN input and get two probabilities for each action (in sum of 1)
        action_prob = model(torch.tensor(obs, dtype=torch.double).to(device))
        # We dont take higher probability but take random value of 0 or 1 based on probabilities from NN
        action = np.random.choice(np.array([0, 1]), p=action_prob.cpu().data.numpy())
        # We take action we got from global NN and take new state, reward and is it terminal state
        new_obs, reward, done, _ = env.step(action)
        if done:
            reward = -20
        ep_reward += reward
        obs = new_obs
        if done:
            all_rewards.append(ep_reward)
            print("Test process " + str(counter.value) + " Episode " + str(ep_num) + " total reward: " + str(
                ep_reward) + " Average reward: " + str(np.mean(all_rewards[-100:])))
            obs = env.reset()
            ep_num += 1
            ep_reward = 21

    if writer is not None:
        writer.close()
    env.close()
