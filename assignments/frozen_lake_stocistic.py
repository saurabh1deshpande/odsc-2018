import gym
import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def main():
    # Starting new environment
    env = gym.make('FrozenLakeNotSlippery-v0')

    # Variable to hold total number of steps and rewards over all the episodes
    # This is required in the end to calculate average number of steps
    # over all the episodes
    total_steps = []
    total_rewards = []
    num_episodes = 1000

    # Observation space for FrozenLake, 16 states: 4x4 Grid
    number_of_states = env.observation_space.n

    # Action Space: 4 Actions Move to North, South, East, West
    # Please refer to https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
    # For more information
    number_of_actions = env.action_space.n

    # Initializing Q table with zeros. Table of 16(rows) x 4(columns)
    # We will track rewards for each state and for each action taken in that state
    Q = torch.zeros([number_of_states,number_of_actions])
    gamma = 0.95
    alpha = 0.9

    for i in range(num_episodes):

        # Start new episode
        state = env.reset()

        steps = 0

        while True:
            steps += 1

            # Choose Action Greedily
            random_values = Q[state] + torch.rand(1,number_of_actions)/1000

            action = torch.max(random_values,1)[1].item()

            # Take action
            new_state, reward, done, info = env.step(action)

            # Update Q table using Q Learning Equation
            # Implement Q Learning Equation

            state = new_state

            # If episode if Done, save total steps taken and total rewards
            # For more information about definition of Done and rewards
            # Please refer to https://gym.openai.com/envs/FrozenLake-v0/
            if done:
                total_steps.append(steps)
                total_rewards.append(reward)
                print("Episode finished after {} steps".format(steps))
                break

    print("% Successful episodes: {0}".format(sum(total_rewards) / num_episodes))
    print("Average Number of Steps:{0}".format(sum(total_steps) / num_episodes))

    plt.figure(figsize=(12, 5))
    plt.title("Total Rewards")
    plt.bar(torch.arange(len(total_rewards)), total_rewards, alpha=0.6, color='green')
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.title("Episod Length")
    plt.bar(torch.arange(len(total_steps)), total_steps, alpha=0.6, color='red')
    plt.show()

if __name__ == "__main__":
    main()



