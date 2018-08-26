import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_bandits
import matplotlib.patches as mpatches


class MultiArmedBandits:

    def __init__(self, k, c, iters,strategy,epsilon=0.5):
        # Number of arms
        self.k = k
        # Exploration parameter
        self.c = c
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 1
        # Step count for each arm
        self.k_n = np.ones(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)
        # Strategy UCB or epsilon
        self.strategy = strategy
        
        # epsilon for epsilon greedy strategy
        self.epsilon = epsilon
        self.env = gym.make('BanditTenArmedGaussian-v0')
        self.env.reset()

    def pull(self):

        a = self.get_action()

        observation, reward, done, info = self.env.step(a)

        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward - self.k_reward[a]) / self.k_n[a]
        
    def get_action(self):
        if self.strategy == 'UCB':
            return np.argmax(self.k_reward + self.c * np.sqrt(
                (np.log(self.n)) / self.k_n))
            
        if self.strategy == 'epsilon':
            explore = np.random.uniform() < self.epsilon

            if explore:
                return self.env.action_space.sample()
            else:
                return np.argmax(self.k_reward)

    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward
            
    def reset(self):
        # Resets results while keeping settings
        self.n = 1
        self.k_n = np.ones(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(self.iters)
        self.k_reward = np.zeros(self.k)
        self.env.reset()            
            
def main():
    k = 10
    iters = 1000
    episodes = 1000
    
    ucb_rewards = run_ucb(k,iters,episodes)
    epsilon_rewards = run_epsilon(k,iters,episodes)
        
    plt.figure(figsize=(12,8))
    plt.plot(ucb_rewards,color='blue')
    plt.plot(epsilon_rewards, color='red')
    plt.legend(bbox_to_anchor=(1.2, 0.5))
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    greedy_patch = mpatches.Patch(color='red', label='epsilon-greedy')
    ucb_patch = mpatches.Patch(color='blue', label='upper confidence bounds')
    plt.legend(handles=[greedy_patch, ucb_patch])
    plt.title("Average Rewards after " 
              + str(episodes) + " Episodes")
    plt.show()

def run_ucb(k,iters,episodes):
    
    
    ucb_rewards = np.zeros(iters)
    # Initialize bandits
    ucb = MultiArmedBandits(k, 2, iters,'UCB')
    
    
    # Run experiments
    for i in range(episodes):
        print(f"Running UCB episod:{i}")
        ucb.reset()
        # Run experiments
        ucb.run()
        
        # Update long-term averages
        ucb_rewards = ucb_rewards + (
            ucb.reward - ucb_rewards) / (i + 1)
    
    return ucb_rewards


def run_epsilon(k,iters,episodes):
    
    
    epsilon_rewards = np.zeros(iters)
    # Initialize bandits
    ucb = MultiArmedBandits(k, 1, iters,'epsilon')
    
    
    # Run experiments
    for i in range(episodes):
        print(f"Running Epsilon episod:{i}")
        ucb.reset()
        # Run experiments
        ucb.run()
        
        # Update long-term averages
        epsilon_rewards = epsilon_rewards + (
            ucb.reward - epsilon_rewards) / (i + 1)
    return epsilon_rewards

if __name__ == "__main__":
    main()