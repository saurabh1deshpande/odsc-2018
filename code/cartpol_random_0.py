#  Importing gym
import gym
import matplotlib.pyplot as plt


def main():
    # Starting new environment
    env = gym.make('CartPole-v1')

    # Variable to hold total number of steps over all the episods
    # This is required in the end to calculate average number of steps
    # over all the episodes
    total_steps = []
    num_episodes = 1000

    for i in range(num_episodes):
        #  As this is a fresh start of an episod,
        #  use reset to start new episod in environment
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

            print(new_state)
            print(info)

            # Gym environment will return done = True
            # if episod is finished
            if done:
                total_steps.append(steps)
                print("Episode finished after {} steps".format(steps))
                break

    #  Close the environment
    env.close()
    env.env.close()

    #  Calculate average number of steps we survived for all the episodes and plot
    print("Average Number of Steps:{0}".format(sum(total_steps)/num_episodes))
    plt.plot(total_steps)
    plt.show()


if __name__ == "__main__":
    main()



