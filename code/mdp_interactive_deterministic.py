import random


class Environment:
    def __init__(self):
        self.current_state = 0


    def reset(self):
        return self.current_state

    def take_action(self,a):
        new_state = None
        reward = None

        if a == 1 and self.current_state == 0:
            new_state = 0
            reward = random.randint(5, 9)

        elif a == 2 and self.current_state == 0:
            new_state = 1
            reward = -1 * random.randint(8, 13)

        elif a == 1 and self.current_state == 1:
            new_state = 0
            reward = random.randint(30, 40)

        elif a == 2 and self.current_state == 1:
            new_state = 1
            reward = random.randint(5, 9)

        self.current_state = new_state
        return new_state, reward

def main():

    state_names = ['A', 'B']

    env = Environment()
    current_state = env.reset()

    print(f'State:{state_names[current_state]}')

    n = 1
    while True:
        action = input(f"Time Step:{n}\t\tState:{state_names[current_state]}\t\tAction:")
        action = int(action)
        new_state, reward = env.take_action(action)
        print(f"Reward:{reward:+}")
        current_state = new_state
        n += 1

if __name__ == "__main__":
    main()