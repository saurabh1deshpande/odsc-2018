import gym
import time
import matplotlib.pyplot as plt



def main():
    env = gym.make('FrozenLake-v0')

    total_steps = []
    num_episods = 1000

    for i in range(num_episods):

        state = env.reset()

        steps = 0

        while True:
            steps += 1

            action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)

            # print(new_state)
            # print(info)
            time.sleep(0.4)
            env.render()

            if done:
                total_steps.append(steps)
                print("Episode finished after {} steps".format(steps))
                break

    print("Average Number of Steps:{0}".format(sum(total_steps)/num_episods))
    plt.plot(total_steps)
    plt.show()


if __name__ == "__main__":
    main()



