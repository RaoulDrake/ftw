from typing import Callable

from ftw.agents.tf.ftw import FTWLearnerStandalone as Learner
from ftw.pbt.arena import Arena
from ftw.pbt.chief import Chief

import acme
from acme import specs

import sonnet as snt
import dm_env


class FTWJobPool:

    def __init__(self,
                 sample_environment_fn: Callable[[], dm_env.Environment],
                 arenas: int = 1,
                 learners: int = 2,
                 team_size: int = 1,
                 unroll_length: int = 20,
                 min_steps: int = 7000,
                 t: int = 200,
                 pixel_observations: bool = True):
        self._sample_environment = sample_environment_fn

        env = self._sample_environment()
        num_env_events = 1
        if hasattr(env, 'num_events'):
            num_env_events = env.num_events

        env_spec = acme.make_environment_spec(env)
        env_spec = specs.EnvironmentSpec(
            observations=env_spec.observations[0],
            actions=env_spec.actions[0],
            rewards=env_spec.rewards[0],
            discounts=env_spec.discounts
        )

        # Create learners
        self.learners = []
        for i in range(learners):
            embed = None
            use_pixel_control = True
            if not pixel_observations:
                # Use an MLP embedding instead of the FTW visual embedding.
                embed = snt.nets.MLP(output_sizes=[256, 256])
                use_pixel_control = False
            self.learners.append(Learner(
                environment_spec=env_spec,
                sequence_length=unroll_length,
                num_environment_events=num_env_events,
                embed=embed,
                use_pixel_cotrol=use_pixel_control,
                agent_id=i
            ))

        # Create arenas
        self.arenas = []
        for i in range(arenas):
            self.arenas.append(Arena(self._sample_environment, label=f'arena_{i}'))

        # Create chief
        self.chief = Chief(learners, team_size, min_steps, t)

        self._all_jobs = self.arenas + self.learners + [self.chief]

        # Attach itself to all jobs
        for job in self._all_jobs:
            job.attach_job_pool(self)

    def get_jobs(self):
        return self._all_jobs
