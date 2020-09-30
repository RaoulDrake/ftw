from acme import wrappers
from ftw import wrappers as marl_wrappers
from ftw.environments import marl_environment as marl_env

import numpy as np

import dm_env
import gym
import ma_gym


class PongDuelPbtEnv(dm_env.Environment, marl_env.MARLEnvironment):

    def __init__(self):
        self._environment = wrappers.wrap_all(
            environment=gym.make('PongDuel-v0'),
            wrappers=[
                marl_wrappers.MarlGymWrapper,
                marl_wrappers.ObservationActionRewardMarlWrapper,
                wrappers.SinglePrecisionWrapper
            ]
        )
        self._episode_return = np.zeros(shape=[2])
        self._last_outcome = 0.5

    def reset(self):
        self._episode_return = np.zeros(shape=[2])
        self._last_outcome = 0.5
        return self._environment.reset()

    def step(self, action):
        timestep = self._environment.step(action)
        for i in range(2):
            self._episode_return[i] += timestep.reward[i]
        if timestep.last():
            if self._episode_return[0] > self._episode_return[1]:
                self._last_outcome = 0.0
            elif self._episode_return[1] > self._episode_return[0]:
                self._last_outcome = 1.0
            else:
                self._last_outcome = 0.5
        return timestep

    def observation_spec(self):
        return self._environment.observation_spec()

    def action_spec(self):
        return self._environment.action_spec()

    def reward_spec(self):
        return self._environment.reward_spec()

    def discount_spec(self):
        return self._environment.discount_spec()

    def get_outcome(self):
        return self._last_outcome
