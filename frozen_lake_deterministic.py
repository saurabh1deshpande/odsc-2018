import gym
import time
import torch
import random
import matplotlib.pyplot as plt

from gym.envs.registration import register
plt.style.use('ggplot')
#register(
#    id='FrozenLakeNotSlippery-v0',
#    entry_point='gym.envs.toy_text:FrozenLakeEnv',
#    kwargs={'map_name' : '4x4', 'is_slippery': False},
#    max_episode_steps=100,
#    reward_threshold=0.78, # optimum = .8196
#)


def main():
    env = gym.make('FrozenLakeNotSlippery-v0')

    total_steps = []
    total_rewards = []
    num_episods = 1000

    number_of_states = env.observation_space.n
    number_of_actions = env.action_space.n
    Q = torch.zeros([number_of_states,number_of_actions])
    gamma = 1

    for i in range(num_episods):

        state = env.reset()

        steps = 0

        while True:
            steps += 1

            random_values =  Q[state] + torch.rand(1,number_of_actions)/1000

            action = torch.max(random_values,1)[1].item()

            new_state, reward, done, info = env.step(action)

            Q[state, action] = reward + gamma * torch.max(Q[new_state])

            state = new_state

            # time.sleep(0.4)
            # env.render()
            #
            # print(new_state)
            # print(info)

            if done:
                total_steps.append(steps)
                total_rewards.append(reward)
                print("Episode finished after {} steps".format(steps))
                break

    print(Q)
    print("% Successful episods: {0}".format(sum(total_rewards)/num_episods))
    print("Average Number of Steps:{0}".format(sum(total_steps)/num_episods))
    
    plt.figure(figsize=(12,5))
    plt.title("Total Rewards")
    plt.bar(torch.arange(len(total_rewards)), total_rewards, alpha = 0.6, color = 'green')
    plt.show()
    
    plt.figure(figsize=(12,5))
    plt.title("Episod Length")
    plt.bar(torch.arange(len(total_steps)), total_steps, alpha = 0.6, color = 'red')
    plt.show()
    
    
    
if __name__ == "__main__":
    main()



