#  Importing gym
import gym


def main():
    # Starting new environment
    env = gym.make('FrozenLake-v0')

    # Variable to hold total number of steps over all the episodes
    # This is required in the end to calculate average number of steps
    # over all the episodes
    total_steps = []
    num_episodes = 100
    total_rewards = 0

    for i in range(num_episodes):
        #  As this is a fresh start of an episode,
        #  use reset to start new episode in environment
        env.reset()

        # steps - variable to store number of steps we survive
        # in this episode
        steps = 0

        #  Try until is episode is finished
        while True:
            steps += 1

            #  Select any random action
            action = env.action_space.sample()

            #  Take that action on environment
            #  and get new state and rewards
            new_state, reward, done, info = env.step(action)

            total_rewards += reward

            print(new_state)
            print(info)

            # Render the current state of the env
            env.render()

            # Gym environment will return done = True
            # if episod is finished
            if done:
                total_steps.append(steps)
                print("Episode finished after {} steps".format(steps))
                break

    # Close the environment
    print(f'Total Rewards:{total_rewards}')
    env.close()
    env.env.close()

    #  Calculate average number of steps we survived for all the episodes
    print("Average Number of Steps:{0}".format(sum(total_steps) / num_episodes))

    print(f'Success Rate:{total_rewards / num_episodes}')


if __name__ == "__main__":
    main()