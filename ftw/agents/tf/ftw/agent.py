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

"""For The Win (FWT) agent implementation."""

from typing import Mapping, List, Union, Tuple,Optional, Sequence
from ftw.types import Initializer

import acme
from acme import specs
from acme import types
from ftw.agents.tf.ftw import utils as ftw_utils
from ftw.agents.tf.ftw import acting
from ftw.agents.tf.ftw import learning as learning
from ftw.tf import hyperparameters as hp
from ftw.tf import internal_reward
from ftw.tf import networks
from ftw.tf.networks import recurrence
from ftw.tf.networks.dnc import access
from ftw.tf.networks import embedding as ftw_embedding
from acme.tf.networks import embedding as acme_embedding
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import dm_env
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf


class FTWWithoutActor:

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
            logger: loggers.Logger = None):
        # TODO: docstring
        if embed is not None and (isinstance(embed, ftw_embedding.OAREmbedding) or
                                  isinstance(embed, acme_embedding.OAREmbedding)):
            raise ValueError(f"FTW class automatically wraps the supplied embed module as an "
                             f"observation_action_reward.OAR embedding, so supplying such an embedding "
                             f"to the constructor of FTW is not supported. Type of supplied embed module: "
                             f"{type(embed)}")
        if not(isinstance(core_type, str) and core_type in ['lstm', 'rpth', 'rpth_dnc']):
            raise ValueError(f"core_type must be a str and one of ['lstm', 'rpth', 'rpth_dnc'] but was "
                             f"{core_type} of type {type(core_type)}")

        num_actions = environment_spec.actions.num_values

        # Internalize some args that are needed for helper functions.
        self._use_pixel_control = use_pixel_cotrol
        self._use_reward_prediction = use_reward_prediction
        self._sequence_length = sequence_length
        self._reward_prediction_sequence_length = reward_prediction_sequence_length
        self._reward_prediction_sequence_period = reward_prediction_sequence_period
        self._uint_pixels_to_float = uint_pixels_to_float

        # Initialize hyperparameters.
        self._hypers = ftw_utils.initialize_hypers(
            slow_core_period_min_max=slow_core_period_min_max,
            slow_core_period_init_value=slow_core_period_init_value,
            learning_rate=learning_rate,
            entropy_cost=entropy_cost,
            reward_prediction_cost=reward_prediction_cost,
            pixel_control_cost=pixel_control_cost,
            kld_prior_fixed_cost=kld_prior_fixed_cost,
            kld_prior_posterior_cost=kld_prior_posterior_cost,
            scale_grads_fast_to_slow=scale_grads_fast_to_slow
        )

        # Initialize internal rewards.
        self._internal_rewards = ftw_utils.initialize_internal_rewards(
            num_events=num_environment_events,
            init_value_or_range=internal_rewards
        )

        # Initialize embedding module, if necessary.
        if embed is None:
            embed = networks.vision.FtwTorso()
        oar_embed = ftw_embedding.OAREmbedding(
            torso=embed, num_actions=num_actions,
            internal_rewards=self._internal_rewards)
        # Make recurrent core.
        if core_type == 'rpth_dnc' or core_type == 'rpth':
            memory = None
            if core_type == 'rpth_dnc':
                memory = access.MemoryAccess(memory_size=dnc_memory_size,
                                             word_size=dnc_word_size,
                                             num_reads=dnc_num_reads)
            core = recurrence.RPTH(
                period=self._hypers['period'].variable,
                hidden_size=hidden_size,
                num_dimensions=num_dimensions,
                dnc_clip_value=dnc_clip_value,
                use_dnc_linear_projection=use_dnc_linear_projection,
                init_scale=init_scale,
                min_scale=min_scale,
                tanh_mean=tanh_mean,
                fixed_scale=fixed_scale,
                use_tfd_independent=use_tfd_independent,
                w_init=variational_unit_w_init,
                b_init=variational_unit_b_init,
                shared_memory=memory,
                strict_period_order=strict_period_order,
                scale_gradients_fast_to_slow=self._hypers['scale_grads_fast_to_slow'].variable
            )
        else:  # 'lstm'
            core = snt.LSTM(hidden_size)
        # Make policy network
        self._policy_network = networks.FtwNetwork(
            embed=oar_embed, core=core, num_actions=num_actions,
            head_hidden_size=hidden_size
        )
        # Make Pixel control network, if enabled.
        pixel_control_network = None
        if use_pixel_cotrol:
            core_for_pixel_control = core
            if isinstance(core, recurrence.RPTH):
                core_for_pixel_control = networks.RPTHZWrapper(rpth_core=core_for_pixel_control)
            pixel_control_network = networks.RNNPixelControlNetwork(
                embed=oar_embed, core=core_for_pixel_control,
                num_actions=num_actions)
        # Make Reward prediction network, if enabled
        reward_prediction_network = None
        if use_reward_prediction:
            # Reward prediction doesn't use OAR wrapper, so we need to supply embed instead of oar_embed.
            reward_prediction_network = networks.RewardPredictionNetwork(
                embed=embed, hidden_size=128)

        # Create logger.
        self._logger = logger or loggers.TerminalLogger(f'agent_{agent_id}')

        # Create reverb tables & reverb server.
        tables, self._can_sample_queue, can_sample_auxiliary = ftw_utils.create_reverb_tables(
            batch_size=batch_size, max_queue_size=max_queue_size,
            use_pixel_control=self._use_pixel_control,
            use_reward_prediction=self._use_reward_prediction
        )
        self._server = reverb.Server(tables, port=None)
        self._replay_server_address = f'localhost:{self._server.port}'

        # DATASETS
        # Additional dataset object(s) to learn from.
        extra_spec = {
            'core_state': self._policy_network.initial_state(1),
            'logits': tf.ones(shape=(1, num_actions), dtype=tf.float32)
        }
        # Remove batch dimensions.
        extra_spec = tf2_utils.squeeze_batch_dim(extra_spec)
        learner_client = reverb.TFClient(self._replay_server_address)
        # Create datasets
        (queue_dataset, pixel_control_dataset, nonzero_reward_prediction_dataset,
         zero_reward_prediction_dataset) = ftw_utils.create_datasets(
            learner_client=learner_client,
            environment_spec=environment_spec,
            batch_size=batch_size,
            sequence_length=sequence_length,
            extra_spec=extra_spec,
            use_pixel_control=self._use_pixel_control,
            use_reward_prediction=self._use_reward_prediction,
            reward_prediction_sequence_length=reward_prediction_sequence_length
        )

        # Create network variables.
        tf2_utils.create_variables(self._policy_network, [environment_spec.observations])
        if pixel_control_network is not None:
            tf2_utils.create_variables(pixel_control_network, [environment_spec.observations])
        if reward_prediction_network is not None:
            tf2_utils.create_variables(reward_prediction_network, [
                environment_spec.observations.observation.replace(
                    shape=(reward_prediction_sequence_length,) + environment_spec.observations.observation.shape
                )])

        # Create learner.
        self._learner = learning.FtwLearner(
            policy_network=self._policy_network,
            dataset=queue_dataset,
            learning_rate=self._hypers['learning_rate'].variable,
            slow_core_period=self._hypers['period'].variable,
            internal_rewards=self._internal_rewards,
            pixel_control_network=pixel_control_network,
            reward_prediction_network=reward_prediction_network,
            pixel_control_dataset=pixel_control_dataset,
            nonzero_reward_prediction_dataset=nonzero_reward_prediction_dataset,
            zero_reward_prediction_dataset=zero_reward_prediction_dataset,
            entropy_cost=self._hypers['entropy_cost'].variable,
            kld_prior_fixed_cost=self._hypers['kld_prior_fixed_cost'].variable,
            kld_prior_posterior_cost=self._hypers['kld_prior_posterior_cost'].variable,
            pixel_control_cost=self._hypers['pixel_control_cost'].variable,
            reward_prediction_cost=self._hypers['reward_prediction_cost'].variable,
            baseline_cost=baseline_cost,
            discount=discount,
            max_abs_reward=max_abs_reward,
            max_gradient_norm=max_gradient_norm,
            rms_prop_epsilon=rms_prop_epsilon,
            learning_rate_decay_steps=learning_rate_decay_steps,
            can_sample=self._can_sample_queue,
            can_sample_auxiliary=can_sample_auxiliary,
            uint_pixels_to_float=uint_pixels_to_float,
            learner_id=agent_id,
            counter=counter,
            logger=logger)

    def get_step(self):
        # TODO: docstring
        return self._learner.get_step()

    def get_weights(self) -> Mapping[str, List[tf.Variable]]:
        # TODO: docstring
        return self._learner.get_weights()

    def get_hypers(self) -> hp.HyperparametersContainer:
        return hp.HyperparametersContainer(self._hypers)

    def get_rewards(self) -> internal_reward.InternalRewards:
        return self._internal_rewards

    def set_step(self, step):
        # TODO: docstring
        self._learner.set_step(step)

    def set_weights(self, weights: Mapping[str, List[tf.Variable]]):
        # TODO: docstring
        self._learner.set_weights(weights)

    def set_hypers(self, other_hypers: hp.HyperparametersContainer):
        # TODO: docstring
        self.get_hypers().set(other_hypers.get())

    def set_rewards(self, other_rewards: internal_reward.InternalRewards):
        # TODO: docstring
        self._internal_rewards.set(other_rewards.get())


class FTW(acme.Actor, FTWWithoutActor):

    def __init__(
            self,
            environment_spec: specs.EnvironmentSpec,
            sequence_length: int,
            num_environment_events: int = 1,
            embed: snt.Module = None,
            max_queue_size: int = 32,
            batch_size: int = 16,
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
            logger: loggers.Logger = None):
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

        # Create adders for actor.
        adder, rp_adder = ftw_utils.create_adders(
            server_address=self._replay_server_address,
            sequence_length=sequence_length,
            use_pixel_control=self._use_pixel_control,
            use_reward_prediction=self._use_reward_prediction,
            reward_prediction_sequence_length=reward_prediction_sequence_length,
            reward_prediction_sequence_period=reward_prediction_sequence_period
        )

        # Create actor and learner.
        self._actor = acting.FtwActor(
            network=self._policy_network, adder=adder, reward_prediction_adder=rp_adder,
            uint_pixels_to_float=uint_pixels_to_float)

    def observe_first(self, timestep: dm_env.TimeStep):
        self._actor.observe_first(timestep)

    def observe(
            self,
            action: types.NestedArray,
            next_timestep: dm_env.TimeStep,
    ):
        self._actor.observe(action, next_timestep)

    def update(self):
        # Run a number of learner steps (usually gradient steps).
        while self._can_sample_queue():
            self._learner.step()

    def select_action(self, observation: np.ndarray) -> int:
        return self._actor.select_action(observation)

    def run(self):
        # TODO: docstring
        while True:
            self.update()
