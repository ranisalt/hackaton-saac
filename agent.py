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
        self.done = False

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
    from neupy import algorithms, layers

    walker = SillyWalker()

    for i in range(50):
        print('generating data:', i)
        while not walker.done:
            walker.step()
        walker.reset()

    x = np.array([list(s) + [r] for s, r in zip(walker.state_history[:-1],
                                                walker.reward_history)])

    y = np.array(walker.action_history)

    print(x.shape, y.shape)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=(90 / 100),
    )

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    def check_goal(goal):
        def callback(net):
            if net.errors.last() < goal:
                raise StopTraining("Goal reached")

        return callback

    net = algorithms.MinibatchGradientDescent(
        [
            layers.Input(15),
            layers.Relu(10),
            layers.Relu(4),
        ],
        error='mse',
        step=0.1,
        verbose=True,
        show_epoch='10 times',
        epoch_end_signal=check_goal(0.01),
    )

    net.architecture()

    net.train(x_train, y_train, x_test, y_test, epochs=1000)

    walker.reset()

    while not walker.done:
        s = np.array([list(walker.state) + [1]])
        prediction = net.predict(s)[0]

        print(prediction)

        action = Action(*prediction)
        walker.step(action)
        walker._env.render()
