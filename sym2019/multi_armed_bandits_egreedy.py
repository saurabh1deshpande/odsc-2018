import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_bandits
import matplotlib.patches as mpatches


def main():
    # Number of bandits
    num_of_bandits = 10

    # For each episode we will run these many iterations
    iterations = 500
    episodes = 400

    epsilons = [{
        'val':0.0,
        'color':'red'
    },{
        'val':0.01,
        'color':'blue'
    },{
        'val':0.1,
        'color':'yellow'
    },{
        'val':0.5,
        'color':'green'
    },{
        'val':1.0,
        'color':'black'
    },]

    # Create environment - Gaussian Distribution
    env = gym.make('BanditTenArmedGaussian-v0')
    #env = gym.make('BanditTenArmedRandomRandom-v0')
    #env = gym.make('BanditTenArmedRandomFixed-v0')
    #env = gym.make('BanditTenArmedUniformDistributedReward-v0')

    # Run all episodes

    for epsilon in epsilons:
        epsilon['rewards'] = run_epsilon(env, num_of_bandits, iterations, episodes,epsilon.get('val'))
        
    plt.figure(figsize=(12, 8))

    handles = []
    for epsilon in epsilons:
        plt.plot(epsilon.get('rewards'), color=epsilon.get('color'))
        handles.append(mpatches.Patch(color=epsilon.get('color'), label=f'{epsilon.get("val")}'))
    plt.legend(bbox_to_anchor=(1.2, 0.5))
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.legend(handles=handles)
    plt.title("Average Rewards after "
              + str(episodes) + " Episodes")
    plt.show()


def run_epsilon(env, num_of_bandits, iterations, episodes,epsilon):
    """
    This method will run all the episodes with epsilon greedy strategy
    :param env: Bandit Gym Environment
    :param num_of_bandits: Number of bandit arms
    :param iterations: Iterations per episode
    :param episodes: Number of episodes
    :return: Array of length equal to number of episodes having mean reward per episode
    """

    # Initialize total mean rewards array per episode by zero
    print(f"Running for epsilon:{epsilon}")
    epsilon_rewards = np.zeros(iterations)

    for i in range(episodes):

        n = 1
        action_count_per_bandit = np.ones(num_of_bandits)
        mean_reward = 0
        total_rewards = np.zeros(iterations)
        mean_reward_per_bandit = np.zeros(num_of_bandits)
        env.reset()

        for j in range(iterations):
            a = get_epsilon_action(epsilon, env, mean_reward_per_bandit)

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

        # Update mean episode rewards once all the iterations of the episode are done
        epsilon_rewards = epsilon_rewards + (total_rewards - epsilon_rewards) / (i + 1)

    return epsilon_rewards


def get_epsilon_action(epsilon, env, mean_reward_per_bandit):
    """
    This method will return action by epsilon greedy
    :param epsilon: Parameter for Greedy Strategy, exploration vs exploitation
    :param env: Gym environment to select random action (Exploration)
    :param mean_reward_per_bandit: Mean reward per bandit for selecting greedily (Exploitation)
    :return:
    """
    explore = np.random.uniform() < epsilon

    if explore:
        return env.action_space.sample()
    else:
        return np.argmax(mean_reward_per_bandit)


if __name__ == "__main__":
    main()