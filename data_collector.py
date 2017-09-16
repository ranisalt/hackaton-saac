import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from itertools import count

import gym
from gym import wrappers


def save(actions, observations, rewards, name):
    data = {
        'actions': [list(a) for a in actions],
        'observations': [list(a) for a in observations],
        'rewards': rewards,
    }

    with open(f'results/{name}.json', 'w') as file:
        json.dump(data, file)


def iteration(env):
    env.reset()

    done = False

    while not done:
        action = env.action_space.sample()

        observation, reward, done, _ = env.step(action)

        yield action, observation, reward

# env = gym.wrappers.Monitor(gym.make('BipedalWalker-v2'),
#                            'results', force=True)


envs = defaultdict(lambda: gym.make('BipedalWalker-v2'))


def proc(i):
    print('starting process:', i)

    pid = os.getpid()

    env = envs[pid]

    env.reset()

    iteration_data = list(zip(*iteration(env)))

    save(*iteration_data, i)


if __name__ == '__main__':
    with mp.Pool(8) as pool:
        list(pool.imap(proc, count(), 10))
