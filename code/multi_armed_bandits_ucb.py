import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_bandits
import matplotlib.patches as mpatches


def main():
    # Number of bandits
    num_of_bandits = 10

    # For each episode we will run these many iterations
    iterations = 850
    episodes = 1000

    # Create environment - Gaussian Distribution
    env = gym.make('BanditTenArmedGaussian-v0')

    # Run all episodes
    ubc_rewards = run_ucb(env, num_of_bandits,iterations,episodes)
        
    plt.figure(figsize=(12, 8))
    plt.plot(ubc_rewards, color='blue')
    plt.legend(bbox_to_anchor=(1.2, 0.5))
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    greedy_patch = mpatches.Patch(color='blue', label='Upper Confidence Bounds')
    plt.legend(handles=[greedy_patch])
    plt.title("Average Rewards after "
              + str(episodes) + " Episodes")
    plt.show()


def run_ucb(env, num_of_bandits, iterations, episodes):
    """
    This method will run all the episodes with Upper Confidence Bound greedy strategy
    :param env: Bandit Gym Environment
    :param num_of_bandits: Number of bandit arms
    :param iterations: Iterations per episode
    :param episodes: Number of episodes
    :return: Array of length equal to number of episodes having mean reward per episode
    """

    # Initialize total mean rewards array per episode by zero
    ubc_rewards = np.zeros(iterations)
    
    for i in range(episodes):
        print(f"Running UCB episode:{i}")

        n = 1
        action_count_per_bandit = np.ones(num_of_bandits)
        mean_reward = 0
        total_rewards = np.zeros(iterations)
        mean_reward_per_bandit = np.zeros(num_of_bandits)
        env.reset()
        c = 1

        for j in range(iterations):
            a = get_ucb_action(mean_reward_per_bandit, c, n, action_count_per_bandit)

            observation, reward, done, info = env.step(a)

            # Update counts
            n += 1
            action_count_per_bandit[a] += 1

            # Update mean rewards
            mean_reward = mean_reward + (
                    reward - mean_reward) / n

            # Update mean rewards per bandit
            mean_reward_per_bandit[a] = mean_reward_per_bandit[a] + (
                    reward - mean_reward_per_bandit[a]) / action_count_per_bandit[a]

            # Capture mean rewards per iteration
            total_rewards[j] = mean_reward

        ubc_rewards = ubc_rewards + (total_rewards - ubc_rewards) / (i + 1)

    return ubc_rewards


def get_ucb_action(mean_reward_per_bandit, c, n, action_count_per_bandit):
    return np.argmax(mean_reward_per_bandit + c * np.sqrt(
        (np.log(n)) / action_count_per_bandit))

if __name__ == "__main__":
    main()