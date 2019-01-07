
def main():

    # Number of bandits
    num_of_bandits = 10

    # For each episode we will run these many iterations
    iterations = 850
    episodes = 1000

    # Create environment - Gaussian Distribution
    env = gym.make('BanditTenArmedGaussian-v0')

    # Run all episodes
    ucb_rewards = run_mab('ucb',env, num_of_bandits,iterations,episodes)
    epsilon_rewards = run_mab('egreedy',env, num_of_bandits, iterations, episodes)

    plt.figure(figsize=(12, 8))
    plt.plot(ucb_rewards, color='blue')
    plt.plot(epsilon_rewards, color='red')
    plt.legend(bbox_to_anchor=(1.2, 0.5))
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    greedy_patch = mpatches.Patch(color='red', label='epsilon-greedy')
    ucb_patch = mpatches.Patch(color='blue', label='Upper Confidence Bounds')
    plt.legend(handles=[greedy_patch, ucb_patch])
    plt.title("Average Rewards after "
              + str(episodes) + " Episodes")
    plt.show()


def run_mab(strategy, env, num_of_bandits, iterations, episodes):
    """
    This method will run all the episodes as per the strategy passed
    :param strategy: Strategy to select action - 'egreedy' or 'ucb' (Upper Confidence Bound)
    :param env: Bandit Gym Environment
    :param num_of_bandits: Number of bandit arms
    :param iterations: Iterations per episode
    :param episodes: Number of episodes
    :return: Array of length equal to number of episodes having mean reward per episode
    """

    # Initialize total mean rewards array per episode by zero
    mab_rewards = np.zeros(iterations)
    
    for i in range(episodes):
        print(f"Running {strategy.upper()} episode:{i}")

        n = 1
        action_count_per_bandit = np.ones(num_of_bandits)
        mean_reward = 0
        total_rewards = np.zeros(iterations)
        mean_reward_per_bandit = np.zeros(num_of_bandits)
        env.reset()
        c = 2
        epsilon = 0.1

        for j in range(iterations):
            a = get_rbc_action(mean_reward_per_bandit, c, n, action_count_per_bandit) if strategy == 'ucb' else get_epsilon_action(epsilon, env, mean_reward_per_bandit)

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

        mab_rewards = mab_rewards + (total_rewards - mab_rewards) / (i + 1)

    return mab_rewards


def get_rbc_action(mean_reward_per_bandit, c, n, action_count_per_bandit):
    return np.argmax(mean_reward_per_bandit + c * np.sqrt(
        (np.log(n)) / action_count_per_bandit))


def get_epsilon_action(epsilon, env, action_count_per_bandit):
    explore = np.random.uniform() < epsilon

    if explore:
        return env.action_space.sample()
    else:
        return np.argmax(action_count_per_bandit)

if __name__ == "__main__":
    main()