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

"""Network for FTW agent."""

from typing import Tuple, Sequence

from acme.tf.networks import base
from ftw.tf.networks import policy_value as ftw_policy_value
from acme.wrappers import observation_action_reward

import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions

Images = tf.Tensor
QValues = tf.Tensor
Logits = tf.Tensor
Value = tf.Tensor
Embedding = tf.Tensor
CoreOutput = Tuple[tf.Tensor, Sequence[Tuple[tf.Tensor, tf.Tensor]]]


class FtwNetwork(snt.RNNCore):

    def __init__(self,
                 embed: base.Module,
                 core: snt.RNNCore,
                 num_actions: int,
                 head_hidden_size: int = 256,
                 name='simple_ftw_network'):
        super().__init__(name=name)
        self._embed = embed
        self._core = core
        self._head = ftw_policy_value.PolicyValueHead(
            num_actions=num_actions, hidden_size=head_hidden_size)

    def initial_state(self, batch_size: int, **kwargs):
        return self._core.initial_state(batch_size, **kwargs)

    def __call__(
            self, inputs: observation_action_reward.OAR,
            state: snt.LSTMState) -> Tuple[Tuple[Logits, Value, CoreOutput], snt.LSTMState]:
        embeddings = self._embed(inputs)
        core_output, new_state = self._core(embeddings, state)
        logits, value = self._head(core_output.z)  # [B, A]

        return (logits, value, core_output), new_state

    def unroll(
            self,
            inputs: observation_action_reward.OAR,
            states: snt.LSTMState
    ) -> Tuple[Tuple[Logits, Value, CoreOutput], snt.LSTMState]:
        """Efficient unroll that applies embeddings, MLP, & convnet in one pass."""
        embeddings = snt.BatchApply(self._embed)(inputs)  # [T, B, D+A+1]
        core_output, new_states = snt.static_unroll(self._core, embeddings, states)
        logits, values = snt.BatchApply(self._head)(core_output.z)

        return (logits, values, core_output), new_states

    def select_action(self, batched_observation, state):
        # Forward.
        (logits, _, _), new_state = self(batched_observation, state)

        action = tfd.Categorical(logits).sample()

        return action, logits, new_state
