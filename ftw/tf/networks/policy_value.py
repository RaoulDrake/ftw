from typing import Tuple, Union, Sequence

import sonnet as snt
import tensorflow as tf


class PolicyValueHead(snt.Module):
    """A network with linear layers, for policy and value respectively."""

    def __init__(self, num_actions: int, hidden_size: int = 256, activation=tf.nn.relu):
        """Initializes the PolicyValueHead module.

        Args:
            num_actions: Number of actions in discrete action space.
            hidden_size: Size of hidden layers (between input and output layers).
            activation: Activation function to be used by this module (between hidden and output layers).

        Raises:
            ValueError: If shapes and/or types of constructor arguments do not match expected shapes and types.
        """
        super().__init__(name='policy_value_network')
        if not (isinstance(num_actions, int)):
            raise ValueError(f"num_actions must be of type int, but was of type {type(num_actions)}.")
        # TODO: implement decomposed PolicyValueHead
        self._policy_layer = snt.Sequential([
            snt.Linear(hidden_size),
            activation,
            snt.Linear(num_actions)
        ])
        self._value_layer = snt.Sequential([
            snt.Linear(hidden_size),
            activation,
            snt.Linear(1)
        ])

    def __call__(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Returns a (Logits, Value) tuple.

        Args:
            inputs: Hidden state tensor.

        Returns:
            Tuple (logits, value).
        """
        logits = self._policy_layer(inputs)  # [B, A]
        value = tf.squeeze(self._value_layer(inputs), axis=-1)  # [B]

        return logits, value
