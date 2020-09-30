from typing import Sequence, Optional, Callable

import time
import itertools

from acme import core
from acme.utils import counting
from acme.utils import loggers

from ftw.pbt import jobs

import dm_env
import numpy as np


class Arena(jobs.Job, core.Worker):

    def __init__(self,
                 sample_environment_fn: Callable[[], dm_env.Environment],
                 counter: counting.Counter = None,
                 logger: loggers.Logger = None,
                 label: str = 'environment_loop'):
        self._sample_environment = sample_environment_fn
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(label)

    def run(self, num_episodes: Optional[int] = None):
        """Perform the run loop.

        Run the environment loop for `num_episodes` episodes. Each episode is itself
        a loop which interacts first with the environment to get an observation and
        then give that observation to the agent in order to retrieve an action. Upon
        termination of an episode a new episode will be started. If the number of
        episodes is not given then this will interact with the environment
        infinitely.

        Args:
          num_episodes: number of episodes to run the loop for. If `None` (default),
            runs without limit.
        """

        iterator = range(num_episodes) if num_episodes else itertools.count()

        for _ in iterator:
            players = self._job_pool.chief.get_match()

            actors = [self._job_pool.learners[player].make_actor() for player in players]

            # Create an instance of randomised environment
            env = self._sample_environment()

            # Reset any counts and start the environment.
            start_time = time.time()
            episode_steps = 0
            episode_returns = np.zeros(shape=(len(actors))).astype(np.float32)  # [0] * len(actors)
            timestep = env.reset()
            individual_timesteps = []

            # Make the first observation.
            for i in range(len(actors)):
                individual_timestep = dm_env.TimeStep(
                    step_type=timestep.step_type,
                    reward=timestep.reward[i] if timestep.reward is not None else timestep.reward,
                    discount=timestep.discount,
                    observation=timestep.observation[i]
                )
                individual_timesteps.append(individual_timestep)
                actors[i].observe_first(individual_timestep)

            # Run an episode.
            while not timestep.last():
                # Generate an action from the agent's policy and step the environment.
                actions = []
                for i in range(len(actors)):
                    action = actors[i].select_action(
                        individual_timesteps[i].observation)
                    actions.append(action)

                actions = tuple(actions)
                timestep = env.step(actions)
                individual_timesteps = []
                for i in range(len(actors)):
                    individual_timestep = dm_env.TimeStep(
                        step_type=timestep.step_type,
                        reward=timestep.reward[i] if timestep.reward is not None else timestep.reward,
                        discount=timestep.discount,
                        observation=timestep.observation[i]
                    )
                    individual_timesteps.append(individual_timestep)

                    # Have the agent observe the timestep and let the actor update itself.
                    actors[i].observe(actions[i], next_timestep=individual_timestep)
                    actors[i].update()

                    # Individual book-keeping.
                    episode_returns[i] += individual_timestep.reward

                # Collective book-keeping.
                episode_steps += 1

            # Record counts.
            counts = self._counter.increment(episodes=1, steps=episode_steps)

            # Collect the results and combine with counts.
            steps_per_second = episode_steps / (time.time() - start_time)
            result = {
                'episode_length': episode_steps,
                'episode_returns': episode_returns,
                'steps_per_second': steps_per_second,
            }
            result.update(counts)

            # Log the given results.
            self._logger.write(result)

            # After episode ends, send the outcome to the chief
            outcome = 0.5
            if not hasattr(env, 'get_outcome'):
                print(f"WARNING: Environment {env} has no method get_outcome()."
                      f"Arena will send draw as result to Chief.")
            else:
                outcome = env.get_outcome()
            self._job_pool.chief.process_result(players, outcome)


class MARLEnvironmentLoop(core.Worker):

    def __init__(
            self,
            environment: dm_env.Environment,
            actors: Sequence[core.Actor],
            counter: counting.Counter = None,
            logger: loggers.Logger = None,
            label: str = 'environment_loop',
    ):
        # Internalize agents and environment.
        self._environment = environment
        self._actors = actors
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(label)
        self._chief = None

    def run(self, num_episodes: Optional[int] = None):
        """Perform the run loop.

        Run the environment loop for `num_episodes` episodes. Each episode is itself
        a loop which interacts first with the environment to get an observation and
        then give that observation to the agent in order to retrieve an action. Upon
        termination of an episode a new episode will be started. If the number of
        episodes is not given then this will interact with the environment
        infinitely.

        Args:
          num_episodes: number of episodes to run the loop for. If `None` (default),
            runs without limit.
        """

        iterator = range(num_episodes) if num_episodes else itertools.count()

        for _ in iterator:
            # Reset any counts and start the environment.
            start_time = time.time()
            episode_steps = 0
            episode_returns = np.zeros(shape=(len(self._actors))).astype(np.float32)  # [0] * len(self._actors)
            timestep = self._environment.reset()
            individual_timesteps = []

            # Make the first observation.
            for i in range(len(self._actors)):
                individual_timestep = dm_env.TimeStep(
                    step_type=timestep.step_type,
                    reward=timestep.reward[i] if timestep.reward is not None else timestep.reward,
                    discount=timestep.discount,
                    observation=timestep.observation[i]
                )
                individual_timesteps.append(individual_timestep)
                self._actors[i].observe_first(individual_timestep)

            # Run an episode.
            while not timestep.last():
                # Generate an action from the agent's policy and step the environment.
                actions = []
                for i in range(len(self._actors)):
                    action = self._actors[i].select_action(
                        individual_timesteps[i].observation)
                    actions.append(action)

                actions = tuple(actions)
                timestep = self._environment.step(actions)
                individual_timesteps = []
                for i in range(len(self._actors)):
                    individual_timestep = dm_env.TimeStep(
                        step_type=timestep.step_type,
                        reward=timestep.reward[i] if timestep.reward is not None else timestep.reward,
                        discount=timestep.discount,
                        observation=timestep.observation[i]
                    )
                    individual_timesteps.append(individual_timestep)

                    # Have the agent observe the timestep and let the actor update itself.
                    self._actors[i].observe(actions[i],
                                            next_timestep=individual_timestep)
                    self._actors[i].update()

                    # Individual book-keeping.
                    episode_returns[i] += individual_timestep.reward

                # Collective book-keeping.
                episode_steps += 1

            # Record counts.
            counts = self._counter.increment(episodes=1, steps=episode_steps)

            # Collect the results and combine with counts.
            steps_per_second = episode_steps / (time.time() - start_time)
            result = {
                'episode_length': episode_steps,
                'episode_returns': episode_returns,
                'steps_per_second': steps_per_second,
            }
            result.update(counts)

            # Log the given results.
            self._logger.write(result)
