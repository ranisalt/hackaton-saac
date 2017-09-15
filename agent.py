import gym

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')
    env.reset()

    for _ in range(500):
        env.render()

