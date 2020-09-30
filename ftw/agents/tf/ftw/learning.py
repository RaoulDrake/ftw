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

import time
from typing import Dict, List, Mapping, Optional
import types
from ftw.types import FloatValueOrTFVariable, IntValueOrTFVariable

import acme
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
from ftw.tf.losses import impala as impala_losses
from ftw.tf import internal_reward
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree
import trfl

tfd = tfp.distributions


class FtwLearner(acme.Learner, tf2_savers.TFSaveable):
    """Learner for an importance-weighted advantage actor-critic
    with auxiliary tasks and recurrent processing with temporal hierarchy."""

    def __init__(
            self,
            policy_network: snt.RNNCore,
            dataset: tf.data.Dataset,
            learning_rate: FloatValueOrTFVariable,
            slow_core_period: IntValueOrTFVariable,
            internal_rewards: internal_reward.InternalRewards,
            # Arguments for auxiliary tasks:
            pixel_control_network: Optional[snt.RNNCore] = None,
            reward_prediction_network: Optional[snt.Module] = None,
            pixel_control_dataset: Optional[tf.data.Dataset] = None,
            nonzero_reward_prediction_dataset: Optional[tf.data.Dataset] = None,
            zero_reward_prediction_dataset: Optional[tf.data.Dataset] = None,
            # Arguments for Hyperparameters:
            entropy_cost: FloatValueOrTFVariable = 0.,
            kld_prior_fixed_cost: FloatValueOrTFVariable = 1e-4,
            kld_prior_posterior_cost: FloatValueOrTFVariable = 1e-3,
            pixel_control_cost: FloatValueOrTFVariable = 0.01,
            reward_prediction_cost: FloatValueOrTFVariable = 0.1,
            # Arguments for fixed Hyperparameters:
            baseline_cost: float = 0.5,
            discount: FloatValueOrTFVariable = 0.99,
            max_abs_reward: Optional[float] = None,
            max_gradient_norm: Optional[float] = None,
            rms_prop_epsilon: float = 0.01,
            learning_rate_decay_steps: int = 0,
            # Helper/Utility arguments:
            can_sample=None,
            can_sample_auxiliary=None,
            uint_pixels_to_float: bool = True,
            learner_id: int = 0,
            counter: counting.Counter = None,
            logger: loggers.Logger = None,
    ):
        if not hasattr(policy_network, 'unroll'):
            raise ValueError('FtwLearner: Policy Network supplied in args has no unroll() implementation.')
        if (pixel_control_network is not None and
                not hasattr(pixel_control_network, 'pixel_control_unroll')):
            raise ValueError('FtwLearner: Pixel Control Network supplied in args has no unroll() implementation.')
        if ((nonzero_reward_prediction_dataset is None and zero_reward_prediction_dataset is not None) or
                (nonzero_reward_prediction_dataset is not None and zero_reward_prediction_dataset is None)):
            raise ValueError('FtwLearner: only one reward prediction dataset was supplied to args, but two are needed '
                             '(one with nonzero rewards and one with zero rewards).')

        # Set up decaying learning rate (if learning_rate_decay_steps <= 0, no learning rate schedule is applied).
        self._step_counter = tf.Variable(0, trainable=False, name='step_counter',
                                         dtype=tf.int64, shape=())
        self._learning_rate = learning_rate
        self._decayed_learning_rate = tf.Variable(learning_rate, trainable=False,
                                                  name='decayed_learning_rate',
                                                  dtype=tf.float32, shape=())
        self._learning_rate_decay_steps = tf.constant(learning_rate_decay_steps,
                                                      dtype=tf.int64, shape=(),
                                                      name='learning_rate_schedule_steps')

        # Internalise, optimizer, and dataset.
        self._optimizer = snt.optimizers.RMSProp(learning_rate=self._decayed_learning_rate,
                                                 decay=0.99, momentum=0.0,
                                                 epsilon=rms_prop_epsilon)
        self._policy_network = policy_network
        self._policy_variables = policy_network.variables
        self._iterator = iter(dataset)
        # Boolean value / function returning boolean value indicating if samples can be drawn from the replay table.
        # Used by self.run().
        if isinstance(can_sample, types.FunctionType):
            self._can_sample = can_sample
        elif isinstance(can_sample, bool):
            self._can_sample = lambda: can_sample
        else:
            self._can_sample = lambda: True

        # Set up auxiliary tasks.
        if isinstance(can_sample_auxiliary, types.FunctionType):
            self._can_sample_auxiliary = can_sample_auxiliary
        elif isinstance(can_sample_auxiliary, bool):
            self._can_sample_auxiliary = lambda: can_sample_auxiliary
        else:
            self._can_sample_auxiliary = lambda: False

        self._use_pixel_control = pixel_control_dataset is not None and pixel_control_network is not None
        if self._use_pixel_control:
            self._pixel_control_network = pixel_control_network
            self._pixel_control_variables = pixel_control_network.variables
            self._pc_optimizer = snt.optimizers.RMSProp(learning_rate=self._decayed_learning_rate,
                                                        decay=0.99, momentum=0.0,
                                                        epsilon=rms_prop_epsilon)
            self._pixel_control_iterator = iter(pixel_control_dataset)  # pytype: disable=wrong-arg-types

        self._use_reward_prediction = (
                nonzero_reward_prediction_dataset is not None and
                zero_reward_prediction_dataset is not None and
                reward_prediction_network is not None)
        if self._use_reward_prediction:
            self._reward_prediction_network = reward_prediction_network
            self._reward_prediction_variables = reward_prediction_network.variables
            self._rp_optimizer = snt.optimizers.RMSProp(learning_rate=self._decayed_learning_rate,
                                                        decay=0.99, momentum=0.0,
                                                        epsilon=rms_prop_epsilon)
            self._nonzero_reward_prediction_iterator = iter(nonzero_reward_prediction_dataset)
            self._zero_reward_prediction_iterator = iter(
                zero_reward_prediction_dataset)

        # Hyperparameters.
        self._entropy_cost = entropy_cost
        self._pixel_control_cost = pixel_control_cost
        self._reward_prediction_cost = reward_prediction_cost
        self._kld_prior_fixed_cost = kld_prior_fixed_cost
        self._kld_prior_posterior_cost = kld_prior_posterior_cost

        # Fixed Hyperparameters.
        self._baseline_cost = baseline_cost
        self._discount = discount

        # Set up reward/gradient clipping.
        if max_abs_reward is None:
            max_abs_reward = np.inf
        if max_gradient_norm is None:
            max_gradient_norm = 1e10  # A very large number. Infinity results in NaNs.
        self._max_abs_reward = tf.convert_to_tensor(max_abs_reward)
        self._max_gradient_norm = tf.convert_to_tensor(max_gradient_norm)

        # Set up logging/counting.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.TerminalLogger(f'learner_{learner_id}', time_delta=1.)

        self._internal_rewards = internal_rewards

        objects_to_save = {'policy_network': policy_network}
        if self._use_pixel_control:
            objects_to_save['pixel_control_network'] = pixel_control_network
        if self._use_reward_prediction:
            objects_to_save['reward_prediction_network'] = reward_prediction_network
        self._snapshotter = tf2_savers.Snapshotter(
            objects_to_save=objects_to_save, time_delta_minutes=60.,
            directory=f'~/acme/learner_{learner_id}/')

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

        # Internalize arguments needed for methods
        # self._step(), self._pixel_control() and self._reward_prediction().
        self._slow_core_period = slow_core_period
        self._uint_pixels_to_float = uint_pixels_to_float

    @property
    def state(self) -> Mapping[str, tf2_savers.Checkpointable]:
        """Returns the stateful objects for checkpointing."""
        state = {
            'policy_network': self._policy_network,
            'optimizer': self._optimizer,
        }
        if self._use_pixel_control:
            state['pixel_control_network'] = self._pixel_control_network
            state['pixel_control_optimizer'] = self._pc_optimizer
        if self._use_reward_prediction:
            state['reward_prediction_network'] = self._reward_prediction_network
            state['reward_prediction_optimizer'] = self._rp_optimizer
        return state

    def get_step(self):
        return self._step_counter

    def get_weights(self) -> Mapping[str, List[tf.Variable]]:
        return {key: self.state[key].variables for key in self.state}

    def set_step(self, step):
        self._step_counter.assign(step)

    def set_weights(self, weights: Mapping[str, List[tf.Variable]]):
        old_weights = self.get_weights()
        for key in weights:
            try:
                for old, new in zip(old_weights[key], weights[key]):
                    old.assign(new)
            except Exception as exc:
                raise ValueError(f"Exception occurred while trying to set variables for key {key} "
                                 f"of weights dictionary input: {exc}")

    @tf.function
    def _step(self) -> Dict[str, tf.Tensor]:
        """Does an SGD step on a batch of sequences."""

        # Retrieve a batch of data from replay.
        inputs: reverb.ReplaySample = next(self._iterator)
        data = tf2_utils.batch_to_sequence(inputs.data)
        observations, actions, rewards, discounts, extra = (data.observation,
                                                            data.action,
                                                            data.reward,
                                                            data.discount,
                                                            data.extras)

        if self._uint_pixels_to_float:
            observations._replace(
                observation=tf.cast(observations.observation, dtype=tf.float32) / 255.0)

        core_state = tree.map_structure(lambda s: s[0], extra['core_state'])

        actions = actions[:-1]  # [T-1]
        rewards = rewards[:-1]  # [T-1]
        discounts = discounts[:-1]  # [T-1]

        # Transform environment events/rewards.
        rewards = self._internal_rewards.reward(rewards)

        with tf.GradientTape() as tape:
            # Unroll current policy over observations.
            (logits, values, core_outputs), _ = self._policy_network.unroll(
                observations, core_state)  # , observations.observation.shape.as_list()[0])

            # Optionally clip rewards.
            rewards = tf.clip_by_value(rewards,
                                       tf.cast(-self._max_abs_reward, rewards.dtype),
                                       tf.cast(self._max_abs_reward, rewards.dtype))

            # Compute importance sampling weights: current policy / behavior policy. // Critic loss.
            behaviour_logits = extra['logits']
            # with tf.device('/cpu'):
            vtrace_returns = trfl.vtrace_from_logits(
                behaviour_policy_logits=behaviour_logits[:-1],
                target_policy_logits=logits[:-1],
                actions=actions,
                discounts=tf.cast(self._discount * discounts, tf.float32),
                rewards=tf.cast(rewards, tf.float32),
                values=tf.cast(values[:-1], tf.float32),
                bootstrap_value=values[-1],
            )
            critic_loss = impala_losses.compute_baseline_loss(vtrace_returns.vs - values[:-1])

            # Policy-gradient loss.
            policy_gradient_loss = impala_losses.compute_policy_gradient_loss(
                logits=logits[:-1],
                actions=actions,
                advantages=vtrace_returns.pg_advantages
            )

            # Entropy regulariser.
            entropy_loss = impala_losses.compute_entropy_loss(logits[:-1])

            # KL divergence.
            prior_params = core_outputs.distribution_params[0]
            posterior_params = core_outputs.distribution_params[1]

            fixed = tfd.MultivariateNormalDiag(
                loc=tf.zeros_like(prior_params.loc, dtype=prior_params.loc.dtype),
                scale_diag=tf.ones_like(prior_params.scale, dtype=prior_params.scale.dtype) * 0.1)
            prior = tfd.MultivariateNormalDiag(loc=prior_params.loc,
                                               scale_diag=prior_params.scale)
            posterior = tfd.MultivariateNormalDiag(loc=posterior_params.loc,
                                                   scale_diag=posterior_params.scale)

            step_counters = tf.stack([
                    core_state.step + i for i in range(observations.observation.shape.as_list()[0])])
            prior_update_mask = tf.math.floormod(
                step_counters,
                tf.ones_like(step_counters) * self._slow_core_period) == tf.zeros_like(step_counters)
            kld_prior_fixed = tfd.kl_divergence(prior, fixed, name='kld_prior_fixed')  # [T, B]

            kld_prior_fixed_masked = tf.compat.v1.where(
                prior_update_mask,
                kld_prior_fixed, tf.zeros_like(kld_prior_fixed))

            kld_prior_posterior = tfd.kl_divergence(posterior, prior, name='kld_prior_posterior')  # [T, B]

            kld_prior_fixed_loss = tf.reduce_sum(kld_prior_fixed_masked)  # tf.reduce_mean(kld_prior_fixed)
            kld_prior_posterior_loss = tf.reduce_sum(kld_prior_posterior)  # tf.reduce_mean(kld_prior_posterior)

            # Combine weighted sum of actor & critic losses.
            loss = (policy_gradient_loss +
                    self._baseline_cost * critic_loss +
                    self._entropy_cost * entropy_loss +
                    self._kld_prior_fixed_cost * kld_prior_fixed_loss +
                    self._kld_prior_posterior_cost * kld_prior_posterior_loss)

        # Compute gradients and optionally apply clipping.
        gradients = tape.gradient(loss, self._policy_network.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)
        self._optimizer.apply(gradients, self._policy_network.trainable_variables)

        metrics = {
            'loss': loss,
            'critic_loss': self._baseline_cost * critic_loss,
            'entropy_loss': self._entropy_cost * entropy_loss,
            'policy_gradient_loss': policy_gradient_loss,
            'kld_prior_fixed_loss': self._kld_prior_fixed_cost * kld_prior_fixed_loss,
            'kld_prior_posterior_loss': self._kld_prior_posterior_cost * kld_prior_posterior_loss
        }

        return metrics

    @tf.function
    def _pixel_control(self):
        """Does an SGD step on a batch of sequences."""

        # Retrieve a batch of data from replay.
        inputs: reverb.ReplaySample = next(self._pixel_control_iterator)
        data = tf2_utils.batch_to_sequence(inputs.data)

        # Unpack data.
        observations, actions, rewards, discounts, extra = (data.observation,
                                                            data.action,
                                                            data.reward,
                                                            data.discount,
                                                            data.extras)

        if self._uint_pixels_to_float:
            observations._replace(
                observation=tf.cast(observations.observation, dtype=tf.float32) / 255.0)

        core_state = tree.map_structure(lambda s: s[0], extra['core_state'])

        actions = actions[:-1]  # [T-1]

        # Compute Pixel control Q-values and loss.
        with tf.GradientTape() as tape:
            pixel_control_q_vals = self._pixel_control_network.unroll(
                observations, core_state)
            pixel_control_loss, _ = trfl.pixel_control_loss(
                observations=observations.observation, actions=actions,
                action_values=pixel_control_q_vals,
                cell_size=4, discount_factor=0.9, scale=1.0,
                crop_height_dim=(2, 82), crop_width_dim=(2, 82)
            )
            pixel_control_loss = tf.reduce_sum(pixel_control_loss)
            loss = self._pixel_control_cost * pixel_control_loss

        # Compute gradients and optionally apply clipping.
        gradients = tape.gradient(loss, self._pixel_control_network.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)
        self._pc_optimizer.apply(gradients, self._pixel_control_network.trainable_variables)

        metrics = {
            'pixel_control_loss': loss,
        }

        return metrics

    @tf.function
    def _reward_prediction(self):
        """Does an SGD step on a batch of sequences."""

        # Retrieve a batch of data from replay.
        inputs_nonzero: reverb.ReplaySample = next(self._nonzero_reward_prediction_iterator)
        inputs_zero: reverb.ReplaySample = next(self._zero_reward_prediction_iterator)

        # Unpack (OAR) observations and reward.
        data_nonzero = inputs_nonzero.data
        # observations_nonzero, rewards_nonzero = data_nonzero[0], data_nonzero[2]
        observations_nonzero, rewards_nonzero = data_nonzero.observation, data_nonzero.reward

        data_zero = inputs_zero.data
        # observations_zero, rewards_zero = data_zero[0], data_zero[2]
        observations_zero, rewards_zero = data_zero.observation, data_zero.reward

        # Extract labels (last reward in a sequence, per batch element).
        # Semantics of labels: 0 = zero-reward, 1 = negative reward, 2 = positive reward.
        rewards_nonzero = rewards_nonzero[:, -1]
        rewards_nonzero = self._internal_rewards.reward(rewards_nonzero)
        labels_nonzero = tf.compat.v1.where(rewards_nonzero > tf.zeros_like(rewards_nonzero),
                                            tf.ones_like(rewards_nonzero) * 2, rewards_nonzero)
        labels_nonzero = tf.compat.v1.where(labels_nonzero < tf.zeros_like(labels_nonzero),
                                            tf.ones_like(labels_nonzero), labels_nonzero)

        rewards_zero = rewards_zero[:, -1]
        rewards_zero = self._internal_rewards.reward(rewards_zero)
        labels_zero = tf.zeros_like(rewards_zero)

        labels = tf.concat([labels_zero, labels_nonzero], axis=0)
        labels = tf.cast(labels, tf.int32)

        # Concatenate zero-reward and non-zero reward elements into one training batch.
        inputs = tf.concat([observations_zero.observation, observations_nonzero.observation],
                           axis=0)

        if self._uint_pixels_to_float:
            inputs = tf.cast(inputs, dtype=tf.float32) / 255.0

        # Predict and calculate reward prediction (cross entropy) loss.
        with tf.GradientTape() as tape:
            logits = self._reward_prediction_network(inputs)
            rp_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
            rp_loss = tf.reduce_sum(rp_loss)
            loss = self._reward_prediction_cost * rp_loss

        # Compute gradients and optionally apply clipping.
        gradients = tape.gradient(loss, self._reward_prediction_network.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)
        self._rp_optimizer.apply(gradients, self._reward_prediction_network.trainable_variables)

        metrics = {
            'reward_prediction_loss': loss,
        }

        return metrics

    def step(self):
        """Does a step of SGD and logs the results."""

        # Do a batch of SGD.
        results = self._step()

        # Auxiliary tasks.
        if self._can_sample_auxiliary():
            if self._use_pixel_control:
                pc_results = self._pixel_control()
                results.update(pc_results)
            if self._use_reward_prediction:
                rp_results = self._reward_prediction()
                results.update(rp_results)

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        results.update(counts)

        # Increment step counter.
        self._step_counter = self._step_counter.assign_add(1)
        # Update learning rate, if a learning rate decay schedule was specified at construction.
        if self._learning_rate_decay_steps > 0:
            step = tf.minimum(self._step_counter, self._learning_rate_decay_steps)
            self._decayed_learning_rate = self._decayed_learning_rate.assign(
                tf.cast(self._learning_rate, tf.float32) *
                (tf.constant(1, dtype=tf.float32) -
                 (tf.cast(step, tf.float32) / tf.cast(self._learning_rate_decay_steps, tf.float32))))

        # Snapshot and attempt to write logs.
        self._snapshotter.save()
        self._logger.write(results)

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        return [tf2_utils.to_numpy(self._policy_variables)]

    def run(self):
        while True:
            self.step()
