from typing import List, Tuple, Sequence, Callable

from absl.testing import absltest
from absl.testing import parameterized

from ftw.tf.networks import vision

import sonnet as snt
import tensorflow as tf

TEST_CASES = [
    dict(
        testcase_name='MinimalTest',
        batch_size=16,
        observation_shape=[84, 84, 3],
        sequence_len=20,
        conv_filters=((16, 8, 4), (32, 4, 2)),
        residual_filters=((32, 3, 1), (32, 3, 1)),
        hidden_size=256,
        activation=tf.nn.relu,
        activate_last=False
    )
]


class FtwTorsoTest(parameterized.TestCase):

    @parameterized.named_parameters(*TEST_CASES)
    def test_call(self, batch_size: int, observation_shape: List[int],
                  sequence_len: int,
                  conv_filters: Sequence[Tuple[int, int, int]],
                  residual_filters: Sequence[Tuple[int, int, int]],
                  hidden_size: int,
                  activation: Callable[[tf.Tensor], tf.Tensor],
                  activate_last: bool):
        torso = vision.FtwTorso(
            conv_filters=conv_filters,
            residual_filters=residual_filters,
            hidden_size=hidden_size,
            activation=activation,
            activate_last=activate_last)
        dummy_input = tf.random.uniform(shape=[batch_size] + observation_shape,
                                        minval=0.0, maxval=1.0, dtype=tf.float32)
        output = torso(dummy_input)
        self.assertEqual(output.shape, [batch_size, hidden_size])

    @parameterized.named_parameters(*TEST_CASES)
    def test_batch_apply(self, batch_size: int, observation_shape: List[int],
                         sequence_len: int,
                         conv_filters: Sequence[Tuple[int, int, int]],
                         residual_filters: Sequence[Tuple[int, int, int]],
                         hidden_size: int,
                         activation: Callable[[tf.Tensor], tf.Tensor],
                         activate_last: bool):
        torso = vision.FtwTorso(
            conv_filters=conv_filters,
            residual_filters=residual_filters,
            hidden_size=hidden_size,
            activation=activation,
            activate_last=activate_last)
        dummy_input_sequence = tf.random.uniform(
            shape=[sequence_len, batch_size] + observation_shape,
            minval=0.0, maxval=1.0, dtype=tf.float32)
        output = snt.BatchApply(torso)(dummy_input_sequence)
        self.assertEqual(output.shape, [sequence_len, batch_size, hidden_size])


if __name__ == '__main__':
    absltest.main()
