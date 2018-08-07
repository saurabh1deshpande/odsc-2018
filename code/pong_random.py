import gym

def main():
    env = gym.make('PongDeterministic-v4')   
    
    episodes = 3
    
    
    for i in range(1, episodes + 1):
            done = False
            state = env.reset()
            while not done:
                action = env.action_space.sample() # force to choose an action from the network
                state, reward, done, _ = env.step(action)
                env.render()
    
if __name__ == "__main__":
    main()