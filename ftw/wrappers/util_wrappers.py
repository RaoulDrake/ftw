import dm_env
from acme import wrappers
import numpy as np


class ReshapeOneDimFrameStackWrapper(wrappers.base.EnvironmentWrapper):

    def step(self, action) -> dm_env.TimeStep:
        timestep = self._environment.step(action)
        return timestep._replace(observation=np.reshape(
            timestep.observation, np.shape(timestep.observation)[:-1]))

    def reset(self) -> dm_env.TimeStep:
        timestep = self._environment.reset()
        return timestep._replace(observation=np.reshape(
            timestep.observation, np.shape(timestep.observation)[:-1]))

    def observation_spec(self):
        spec = self._environment.observation_spec()
        return spec.replace(shape=spec.shape[:-1])
