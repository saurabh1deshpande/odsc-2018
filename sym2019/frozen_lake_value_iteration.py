#  Importing gym
import gym
import numpy as np
from gym.envs.registration import register

action_map = {
    0: '<',
    1: 'v',
    2: '>',
    3:'^'
}

# Register deterministic FrozenLake environment
# Notice 'is_slippery': False
try:
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4', 'is_slippery': False},
        max_episode_steps=100,
        reward_threshold=0.78, # optimum = .8196
    )
except:
    print("Environment is already registered")

env = gym.make('FrozenLakeNotSlippery-v0')

NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.n
V = np.zeros([NUM_STATES]) # The Value for each state
Pi = np.zeros([NUM_STATES], dtype=int)  # Our policy with we keep updating to get the optimal policy
gamma = 0.9 # discount factor
significant_improvement = 0.01


def best_action_value(s):
    # finds the highest value action (max_a) in state s
    best_a = None
    best_value = float('-inf')

    # loop through all possible actions to find the best current action
    for a in range (0, NUM_ACTIONS):
        env.reset()
        env.env.s = s
        s_new, rew, done, info = env.step(a) #take the action
        v = rew + gamma * V[s_new]
        if v > best_value:
            best_value = v
            best_a = a
    return best_a

def main():
    iteration = 0
    while True:
        # biggest_change is referred to by the mathematical symbol delta in equations
        biggest_change = 0
        for s in range(0, NUM_STATES):
            old_v = V[s]
            action = best_action_value(s)  # choosing an action with the highest future reward
            env.reset()
            env.env.s = s  # goto the state
            s_new, rew, done, info = env.step(action)  # take the action
            V[s] = rew + gamma * V[s_new]  # Update Value for the state using Bellman equation
            Pi[s] = action
            biggest_change = max(biggest_change, np.abs(old_v - V[s]))
        iteration += 1
        print_pi()
        print('\n')
        print(f'Change:{biggest_change:4f}')
        print('*' * 50)
        input()
        if biggest_change < significant_improvement:
            print(iteration, ' iterations done')
            break

    print('\n')

    print_pi()
    input()
    run_episodes(100)

def print_pi():

    for idx, v in enumerate(V):
        if idx % 4 == 0 and idx != 0:
            print('\n')
        print(f'{v:.4}\t', end=" ")

    print('\n')

    for idx, a in enumerate(Pi):
        if idx % 4 == 0 and idx != 0:
            print('\n')
        print(f'{action_map[a]}\t', end=" ")


def run_episodes(num_episodes):
    total_steps = []
    total_rewards = 0
    for i in range(num_episodes):
        state = env.reset()
        steps = 0
        rewards = 0
        while True:
            steps += 1

            action = Pi[state]

            #  Take that action on environment
            #  and get new state and rewards
            new_state, reward, done, info = env.step(action)
            state = new_state

            rewards += reward

            # Gym environment will return done = True
            # if episod is finished
            if done:
                total_steps.append(steps)
                total_rewards += rewards
                print("Episode finished after {} steps".format(steps))
                break

    print(f'Percentage Success:{(total_rewards/num_episodes)*100.0}')

if __name__ == "__main__":
    main()