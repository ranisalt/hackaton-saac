from typing import NamedTuple

import gym
import numpy as np

from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split


class Action(NamedTuple):
    left_hip: float
    left_knee: float
    right_hip: float
    right_knee: float


class State(NamedTuple):
    angle: float
    angular_velocity: float
    x_velocity: float
    y_velocity: float
    left_hip_angle: float
    left_hip_speed: float
    left_knee_angle: float
    left_knee_speed: float
    left_on_ground: float
    right_hip_angle: float
    right_hip_speed: float
    right_knee_angle: float
    right_knee_speed: float
    right_on_ground: float


class SillyWalker:
    def __init__(self):
        self._env = gym.make('BipedalWalker-v2')
        self.action_history = []
        self.state_history = []
        self.reward_history = []

        self.state = self._env.reset()[:14]
        self.done = False

    def random_action(self):
        return Action(*self._env.action_space.sample())

    def step(self, action=None):
        action = action or self.random_action()
        self.action_history.append(action)
        self.state, self.reward, self.done, _ = self._env.step(action)

    def reset(self):
        self._env.reset()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        new_state = new_state[:14]
        self.state_history.append(new_state)
        self._state = State(*new_state)

    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, new_reward):
        self.reward_history.append(new_reward)
        self._reward = new_reward


if __name__ == '__main__':
    walker = SillyWalker()

    # for _ in range(1000):
    while not walker.done:
        walker.step()
        # walker.reset()

    x = np.array(walker.state_history[:-1])
    y = np.array([list(a) + [r] for a, r in zip(walker.action_history,
                                                walker.reward_history)])

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=3/4,
                                                        test_size=1/4)

    model = TPOTRegressor(generations=5, population_size=20, verbosity=2)
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
    model.export('predict.py')
