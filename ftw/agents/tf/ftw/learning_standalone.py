# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Learner for the IMPALA actor-critic agent."""

from typing import Union, Optional, Tuple, List
from ftw.types import Initializer

from acme import specs, types
from acme import core as acme_core
from acme.utils import counting
from acme.utils import loggers

from ftw import core as ftw_core
from ftw.agents.tf.ftw import acting
from ftw.agents.tf.ftw import utils as ftw_utils
from ftw.agents.tf.ftw import agent as ftw_agent
from ftw.pbt import jobs

import sonnet as snt
import tensorflow_probability as tfp

tfd = tfp.distributions


class FTWLearnerStandalone(acme_core.Learner,
                           ftw_agent.FTWWithoutActor,
                           ftw_core.LearnerStandalone,
                           jobs.Job):

    def __init__(
            self,
            environment_spec: specs.EnvironmentSpec,
            sequence_length: int,
            num_environment_events: int = 1,
            embed: snt.Module = None,
            max_queue_size: int = 64,
            batch_size: int = 32,
            hidden_size: int = 256,
            use_pixel_cotrol: bool = True,
            use_reward_prediction: bool = True,
            reward_prediction_sequence_length: int = 3,
            reward_prediction_sequence_period: int = 1,
            # Arguments for recurrent core:
            num_dimensions: int = 256,
            dnc_clip_value=None,
            use_dnc_linear_projection: bool = True,
            init_scale: float = 0.1,
            min_scale: float = 1e-6,
            tanh_mean: bool = False,
            fixed_scale: bool = False,
            use_tfd_independent: bool = False,
            variational_unit_w_init: Optional[Initializer] = None,
            variational_unit_b_init: Optional[Initializer] = None,
            strict_period_order: bool = True,
            dnc_memory_size: int = 450,
            dnc_word_size: int = 32,
            dnc_num_reads: int = 4,
            core_type: str = 'rpth',  # 'rpth_dnc',
            # Arguments for Hyperparameters Initialization:
            slow_core_period_min_max: Tuple[int, int] = (5, 20),
            slow_core_period_init_value: Optional[int] = None,
            learning_rate: Union[float, Tuple[float, float]] = (1e-5, 5 * 1e-3),
            entropy_cost: Union[float, Tuple[float, float]] = (5 * 1e-4, 1e-2),
            reward_prediction_cost: Union[float, Tuple[float, float]] = (0.1, 1.0),
            pixel_control_cost: Union[float, Tuple[float, float]] = (0.01, 0.1),
            kld_prior_fixed_cost: Union[float, Tuple[float, float]] = (1e-4, 0.1),
            kld_prior_posterior_cost: Union[float, Tuple[float, float]] = (1e-3, 1.0),
            scale_grads_fast_to_slow: Union[float, Tuple[float, float]] = (0.1, 1.0),
            internal_rewards: Union[float, Tuple[float, float]] = (0.1, 1.0),
            # Arguments for fixed Hyperparameters:
            baseline_cost: float = 0.5,
            discount: float = 0.99,
            max_abs_reward: float = None,
            max_gradient_norm: float = None,
            rms_prop_epsilon: float = 1e-5,
            learning_rate_decay_steps: int = 0,
            # Helper/Utility arguments:
            uint_pixels_to_float: bool = True,
            agent_id: int = 0,
            counter: counting.Counter = None,
            logger: loggers.Logger = None,):
        # TODO: docstring
        super().__init__(
            environment_spec=environment_spec,
            sequence_length=sequence_length,
            num_environment_events=num_environment_events,
            embed=embed,
            max_queue_size=max_queue_size,
            batch_size=batch_size,
            hidden_size=hidden_size,
            use_pixel_cotrol=use_pixel_cotrol,
            use_reward_prediction=use_reward_prediction,
            reward_prediction_sequence_length=reward_prediction_sequence_length,
            reward_prediction_sequence_period=reward_prediction_sequence_period,
            # Arguments for recurrent core:
            num_dimensions=num_dimensions,
            dnc_clip_value=dnc_clip_value,
            use_dnc_linear_projection=use_dnc_linear_projection,
            init_scale=init_scale,
            min_scale=min_scale,
            tanh_mean=tanh_mean,
            fixed_scale=fixed_scale,
            use_tfd_independent=use_tfd_independent,
            variational_unit_w_init=variational_unit_w_init,
            variational_unit_b_init=variational_unit_b_init,
            strict_period_order=strict_period_order,
            dnc_memory_size=dnc_memory_size,
            dnc_word_size=dnc_word_size,
            dnc_num_reads=dnc_num_reads,
            core_type=core_type,
            # Arguments for Hyperparameters Initialization:
            slow_core_period_min_max=slow_core_period_min_max,
            slow_core_period_init_value=slow_core_period_init_value,
            learning_rate=learning_rate,
            entropy_cost=entropy_cost,
            reward_prediction_cost=reward_prediction_cost,
            pixel_control_cost=pixel_control_cost,
            kld_prior_fixed_cost=kld_prior_fixed_cost,
            kld_prior_posterior_cost=kld_prior_posterior_cost,
            scale_grads_fast_to_slow=scale_grads_fast_to_slow,
            internal_rewards=internal_rewards,
            # Arguments for fixed Hyperparameters:
            baseline_cost=baseline_cost,
            discount=discount,
            max_abs_reward=max_abs_reward,
            max_gradient_norm=max_gradient_norm,
            rms_prop_epsilon=rms_prop_epsilon,
            learning_rate_decay_steps=learning_rate_decay_steps,
            # Helper/Utility arguments:
            uint_pixels_to_float=uint_pixels_to_float,
            agent_id=agent_id,
            counter=counter,
            logger=logger)

    def make_actor(self):
        # TODO: docstring
        adder, rp_adder = ftw_utils.create_adders(
            server_address=self._replay_server_address,
            sequence_length=self._sequence_length,
            use_pixel_control=self._use_pixel_control,
            use_reward_prediction=self._use_reward_prediction,
            reward_prediction_sequence_length=self._reward_prediction_sequence_length,
            reward_prediction_sequence_period=self._reward_prediction_sequence_period
        )
        return acting.FtwActor(
            network=self._policy_network, adder=adder, reward_prediction_adder=rp_adder,
            uint_pixels_to_float=self._uint_pixels_to_float)

    @property
    def replay_server_address(self):
        # TODO: docstring
        return self._replay_server_address

    def step(self):
        self._learner.step()

    def get_variables(self, names: List[str]):
        self._learner.get_variables()

    def run(self):
        self._learner.run()
