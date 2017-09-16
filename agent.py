import pickle

from typing import NamedTuple

import gym


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
        self.done = False

    def clear_history(self):
        self.action_history.clear()
        self.reward_history.clear()
        self.state_history.clear()

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

    def save_history(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.action_history,
                         self.reward_history,
                         self.state_history), f)

    @staticmethod
    def load_history(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
