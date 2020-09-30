from acme.wrappers import observation_action_reward as oar
from acme import wrappers

import dm_env
from dm_env import specs


class MarlGymWrapper(wrappers.GymWrapper):

    def reward_spec(self):
        """Describes the reward returned by the environment.

        By default this is assumed to be a tuple of floats, one for each agent.

        Returns:
          A tuple of `Array` specs, one for each agent.
        """

        return tuple([
            specs.Array(shape=(), dtype=float, name='reward')
            for i in range(self.n_agents)  # self.action_spec())
        ])


class ObservationActionRewardMarlWrapper(oar.ObservationActionRewardWrapper):
    """A wrapper that puts the previous action and reward into the observation of each respective agent.

    Requires that the underlying environment has implemented a public attribute 'n_agents',
    indicating the number of agents of that environment.
    """

    def _augment_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        """Extracts individual observations, actions and rewards from a multi-agent environment TimeStep.

        Constructs OAR NamedTuples for each agent,
        as the original dm-acme OAR wrapper does for the single-agent case.
        """

        oars = tuple([
            oar.OAR(
                observation=timestep.observation[i],
                action=self._prev_action[i],
                reward=self._prev_reward[i])
            for i in range(len(timestep.observation))
        ])
        return timestep._replace(observation=oars)

    def observation_spec(self):
        return tuple([
            oar.OAR(
                observation=self._environment.observation_spec()[i],
                action=self.action_spec()[i],
                reward=self.reward_spec()[i])
            for i in range(self.n_agents)
        ])
