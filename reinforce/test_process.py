import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
from agent_nn import PolicyNN

def test_process(parameters, shared_model_actor, end_flag, wait_queue, episode_counter, wait_first_test):
    # Create enviroment and agent
    env = gym.make(parameters['env_name'])
    writer_test = SummaryWriter(log_dir=parameters['writer_log_dir'] + str(time.time())) if parameters['writer_test'] else None
    # We need to create actor model which will be placeholder for shared actor NN model
    device = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PolicyNN(env.observation_space.shape[0], env.action_space.n).to(device)
    obs = env.reset()
    # We set ep_reward to 21 so we can nullify -20 reward we added at the end, so we can compare results as intended
    ep_reward = 21
    # We will collect every episode reward in this list, but we will only calculate mean of last 100
    all_rewards = []
    print("Starting testing process...")
    # This loop is meant to be over only when we reach mean score of 495 or higher
    while True:
        ep_num = 0
        # Queue for test to get blocked so it doesnt run non-stop but controlled by main
        wait_queue.get()
        # Save current episode since train episode will change while we test
        episode = episode_counter.value
        # Each iteration we need to replace old params with new, update Actor NN params
        model.load_state_dict(shared_model_actor.state_dict())
        # One iteration will play max_test_games episodes/games before updating NN
        while ep_num < parameters['max_test_games']:
            # We send current state as NN input and get two probabilities for each action (in sum of 1)
            action_prob = model(torch.tensor(obs, dtype=torch.double).to(device))
            # We dont take higher probability but take random value of 0 or 1 based on probabilities from NN
            action = np.random.choice(np.array([0, 1]), p=action_prob.cpu().data.numpy())
            # We take action we got from global NN and take new state, reward and is it terminal state
            new_obs, reward, done, _ = env.step(action)
            # To make loss more unrewarding we penalize loss more (instead of default 0)
            if done:
                reward = -20
            # We add current step reward to episode reward
            ep_reward += reward
            # Change new state to be current state so we can continue
            obs = new_obs
            if done:
                # Add episode reward to the list of all episode rewards
                all_rewards.append(ep_reward)
                # We need to reset variables
                obs = env.reset()
                ep_num += 1
                ep_reward = 21
        # After each iteration print results
        if parameters['print_test_results']:
            print("Test process - Episode " + str(episode) + " Average 10 reward: " + str(np.mean(all_rewards[-10:])) + " Average 100 reward: " + str(np.mean(all_rewards[-100:])))
        # Test process needs to send main taht it can start
        if episode == 0:
            wait_first_test.put(0)
        # If we reached necessary goal for 10 episodes test for 90 more for parameters to be valid
        if np.mean(all_rewards[-10:]) >= 495:
            print("Started validating...")
            # Since we only tested each parameters on only 10 episodes (instead of 100 as is requested by Cartpole challenge)
            # We need to test 90 episodes more on same parameters we tested last 10 episodes to make sure they are valid
            while ep_num < 100:
                action_prob = model(torch.tensor(obs, dtype=torch.double).to(device))
                action = np.random.choice(np.array([0, 1]), p=action_prob.cpu().data.numpy())
                new_obs, reward, done, _ = env.step(action)
                if done:
                    reward = -20
                ep_reward += reward
                obs = new_obs
                if done:
                    all_rewards.append(ep_reward)
                    obs = env.reset()
                    ep_num += 1
                    ep_reward = 21

            # If we reached necessary goal and last Actor parameters are valid
            if np.mean(all_rewards[-100:]) >= 495:
                # Since test process is only process that can change end_flag shared variable (and there is only 1 test process)
                # We dont need to use sempahore (lock) and we can just change it for train processes to read it.
                end_flag.value = 1
                print("Testing finished, parameters are valid! Average of last 100 games is " + str(np.mean(all_rewards[-100:])))
                if writer_test is not None:
                    writer_test.add_scalar('mean_reward', np.mean(all_rewards[-100:]), episode)
                break
            else:
                print("Testing finished, parameters are NOT valid! Continuing training...")
        if writer_test is not None:
            writer_test.add_scalar('mean_reward', np.mean(all_rewards[-100:]), episode)
        if episode_counter.value > parameters['max_train_games']:
            break
    # Close enviroment at the end
    if writer_test is not None:
        writer_test.close()
    env.close()
