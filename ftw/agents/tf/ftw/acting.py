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

"""FTW actor implementation."""
from typing import Optional

from acme import adders
from acme import core
from acme import types
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

# from ftw.adders.reverb import rnn_sequence

import tree
import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

from acme.agents.tf.impala.acting import IMPALAActor
import numpy as np


tfd = tfp.distributions


class FtwActor(core.Actor):
    """A recurrent actor."""

    def __init__(
            self,
            network: snt.RNNCore,
            adder: adders.Adder = None,
            reward_prediction_adder: adders.Adder = None,
            variable_client: tf2_variable_utils.VariableClient = None,
            uint_pixels_to_float: bool = True
    ):

        # Store these for later use.
        self._adder = adder
        self._reward_pred_adder = reward_prediction_adder
        self._variable_client = variable_client
        self._network = network

        self._state = None
        self._prev_state = None
        self._prev_logits = None
        self._uint_pixels_to_float = uint_pixels_to_float

    @tf.function
    def _forward_pass(self, batched_observation, state):
        # Forward.
        (logits, _, _), new_state = self._network(batched_observation, state)

        action = tfd.Categorical(logits).sample()

        return action, logits, new_state

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_obs = tf2_utils.add_batch_dim(observation)

        if self._uint_pixels_to_float:
            batched_obs._replace(
                observation=tf.cast(batched_obs.observation, dtype=tf.float32) / 255.0)

        if self._state is None:
            self._state = self._network.initial_state(1)
        action, logits, new_state = self._forward_pass(batched_obs, self._state)

        self._prev_logits = logits
        self._prev_state = self._state
        self._state = new_state

        action = tf2_utils.to_numpy_squeeze(action)

        return action

    def observe_first(self, timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add_first(timestep)
        if self._reward_pred_adder:
            self._reward_pred_adder.add_first(timestep)

        # Set the state to None so that we re-initialize at the next policy call.
        self._state = None

    def observe(
            self,
            action: types.NestedArray,
            next_timestep: dm_env.TimeStep,
    ):
        if (not self._adder) and (not self._reward_pred_adder):
            return
        if self._adder:
            extras = {'logits': self._prev_logits,
                      'core_state': self._prev_state}
            extras = tf2_utils.to_numpy_squeeze(extras)
            self._adder.add(action, next_timestep, extras)
        if self._reward_pred_adder:
            self._reward_pred_adder.add(action, next_timestep)

    def update(self):
        if self._variable_client:
            self._variable_client.update()
