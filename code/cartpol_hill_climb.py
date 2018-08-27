import gym
import torch
import numpy as np
import matplotlib.pyplot as plt


def run_episode(env, parameters):
    """

    :param env: Gym Environment
    :param parameters: Start Parameters for the episode
    :return: Total rewards obtained in this episode
    """
    # Reset / Start fresh episode
    observation = env.reset()
    total_rewards = 0

    # Loop till episode is not finished
    # Please check https://gym.openai.com/envs/CartPole-v1/
    # for definition of episode end
    # done flag will be returned as False of episode is ended
    while True:
        # Compare (dot product) parameters returned by environment
        # with our parameters for the episode. If dot product is
        # less than zero, push cart to left otherwise right
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        total_rewards += reward

        # Terminate either if episode ends or we achieve reward of 200
        if done or total_rewards >= 200:
            break
    return total_rewards


def hill_climb_search():
    """
    Hill Climb Search using noise scaling parameter
    :return: NumPy array of length 4 with random numbers between [-1,1] discounted ny noise scaling
    """
    noise_scaling = 0.5
    return (np.random.rand(4) * 2 - 1) * noise_scaling


def main():
    # Starting new environment
    env = gym.make('CartPole-v1')

    # Variable to hold total rewards for all the trials
    total_rewards = []

    # Total Number of episodes.
    num_episodes = 1000

    # Store best reward and corresponding parameters.
    best_reward = 0
    best_params = None
    counter = 0

    for episode in range(num_episodes):
        counter += 1

        # Get parameter using random search
        parameters = hill_climb_search()

        # Pass environment and parameters to this function
        # which will run the episode till it terminates and return the rewards
        reward = run_episode(env, parameters)

        # Store the rewards for this episode for plotting later
        total_rewards.append(reward)

        # Also update the best reward if the current reward is better then the existing
        if reward > best_reward:
            best_reward = reward
            best_params = parameters
            if reward == 200:
                print('200 achieved on episode {}'.format(episode))

    print("Average Reward after {0} consecutive trails:{1}".format(num_episodes, sum(total_rewards) / num_episodes))
    print("Best Reward:{}".format(best_reward))
    print("Best Params:{}".format(best_params))

    # Plot the rewards
    plt.figure(figsize=(12, 5))
    plt.title("Rewards")
    plt.bar(torch.arange(len(total_rewards)), total_rewards, alpha=0.6, color='green')
    plt.show()

    # Close the environment
    env.close()
    env.env.close()


if __name__ == "__main__":
    main()




