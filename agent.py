import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from itertools import count

import gym
from typing import NamedTuple


class Action(NamedTuple):
    left_hip: float
    left_knee: float
    right_hip: float
    right_knee: float


class Agent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')

    agent = Agent(env.action_space)

    while True:
        ob, reward, done = env.reset(), 0, False

        while not done:
            action = agent.act(ob, reward, done)
            ob, reward, done, *_ = env.step(action)
            env.render()
