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

"""Modules for computing custom embeddings."""

from typing import Union, Sequence, Optional

from acme.tf.networks import base
from acme.wrappers import observation_action_reward

from ftw.tf import internal_reward

import sonnet as snt
import tensorflow as tf


class OAREmbedding(snt.Module):
    """Module for embedding (observation, action, reward) inputs together.

    This module is based on dm-acme's OAREmbedding module, but was enhanced to further support
        - multi-discrete/decomposed action spaces (such as the one from the FTW paper)
        - internal rewards (as used by the FTW agent).

    If a multi-discrete/decomposed action space is used, the action will be embedded as a concatenation of
    one-hot encodings (one encoding per action group in the multi-discrete/decomposed action space).

    If internal rewards are used, the embedding of the reward will be computed
        - in case of scalar original reward and scalar internal reward: as the product between both
        - in case of original rewards vector and scalar internal reward: as the product between both
        - in case original rewards vector and internal rewards vector: as the dot product between both.
    Scalar original reward and internal rewards vector is not a supported use-case.
    """

    def __init__(self, torso: base.Module, num_actions: Union[int, Sequence[int]],
                 internal_rewards: Optional[internal_reward.InternalRewards] = None):
        """Initializes the OAREmbedding module.

        Args:
            torso: Module transforming observations into an embedding
            num_actions: Number of actions in action space. Supports discrete action space (if int is supplied),
                or multi-discrete/decomposed action space (if sequence of ints is supplied, one for each action group).
            internal_rewards: InternalRewards module (as used in the FTW paper). Optional.
                If None, no internal rewards calculation will be done.

        Raises:
            ValueError: If shapes and/or types of constructor arguments do not match expected shapes and types.
        """

        super().__init__(name='oar_embedding')
        num_actions_is_int = isinstance(num_actions, int)
        num_actions_is_int_sequence = (isinstance(num_actions, Sequence) and
                                       all([isinstance(item, int) for item in num_actions]))
        if not (num_actions_is_int or num_actions_is_int_sequence):
            raise ValueError(
                f"num_actions must either be of type int or a sequence of ints (e.g. as a tuple or list). "
                f"num_actions supplied to constructor: {num_actions} (type: {type(num_actions)})")
        if internal_rewards is not None and not isinstance(internal_rewards, internal_reward.InternalRewards):
            raise ValueError(f"internal_rewards must be of type InternalRewards but had type {type(internal_rewards)}")

        self._num_actions = num_actions
        self._internal_rewards = internal_rewards
        if internal_rewards is not None:
            # Variable of internal rewards must be kept as an attribute to be compatible with snapshotting.
            self._internal_rewards_variables = internal_rewards.variable
        self._torso = torso

    def __call__(self, inputs: observation_action_reward.OAR) -> tf.Tensor:
        """Embed each of the (observation, action, reward) inputs & concatenate.

        Args:
            inputs: observation_action_reward.OAR namedtuple containing current observation, last action and
                last reward(s)/events.

        Returns:
            embedding: concatenation of observation, action and reward embeddings.
        """

        # Add dummy trailing dimension to rewards if necessary.
        if len(inputs.reward.shape.dims) == 1:
            inputs = inputs._replace(reward=tf.expand_dims(inputs.reward, axis=-1))

        features = self._torso(inputs.observation)  # [T?, B, D]
        if isinstance(self._num_actions, int):
            action = tf.one_hot(inputs.action, depth=self._num_actions)  # [T?, B, A]
        else:  # decomposed action space
            # create one-hot arrays per action group and concatenate the results to a tf.Tensor
            action = tf.concat(
                [tf.one_hot(action, depth=self._num_actions[i]) for i, action in enumerate(inputs.action)],
                axis=-1)
        reward = inputs.reward
        if self._internal_rewards is not None:
            # If using internal_rewards, reward is dot product between internal_rewards and rewards/events
            reward = self._internal_rewards.reward(events=reward)
        reward = tf.nn.tanh(reward)  # [T?, B, 1]

        embedding = tf.concat([features, action, reward], axis=-1)  # [T?, B, D+A+1]

        return embedding
