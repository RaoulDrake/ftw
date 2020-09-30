import itertools

from absl.testing import absltest
from absl.testing import parameterized

from ftw.tf.networks import vision as ftw_vision
from ftw.tf.networks import auxiliary
from ftw.tf.networks import recurrence
from ftw.tf.networks.dnc import access
from acme.tf.networks import vision, atari

import sonnet as snt
import tensorflow as tf

EMBEDS = [
    'Ftw', 'ResNet', 'Atari'
]

CORES = [
    'RpthDnc', 'RpthLstm', 'Lstm'
]

TEST_PARAMS = itertools.product(EMBEDS, CORES)

TEST_CASES_RNN_PIXEL_CONTROL_NET = [
    dict(
        testcase_name=embed_name + 'Torso' + core_name + 'Core',
        embed_name=embed_name, core_name=core_name
    )
    for embed_name, core_name in TEST_PARAMS
]
TEST_CASES_REWARD_PREDICTION = [
    dict(
        testcase_name=embed_name + 'Torso',
        embed_name=embed_name
    )
    for embed_name in EMBEDS
]


def make_embed(embed_name):
    if embed_name == 'Ftw':
        return ftw_vision.FtwTorso()
    elif embed_name == 'ResNet':
        return vision.ResNetTorso()
    elif embed_name == 'Atari':
        return atari.AtariTorso()


def make_core(core_name):
    if core_name == 'RpthDnc':
        mem = access.MemoryAccess(
            memory_size=128, word_size=32, num_reads=4, num_writes=1)
        return recurrence.RPTHZWrapper(rpth_core=recurrence.RPTH(
            period=5, hidden_size=256, num_dimensions=256,
            shared_memory=mem
        ))
    elif core_name == 'RpthLstm':
        return recurrence.RPTHZWrapper(rpth_core=recurrence.RPTH(
            period=5, hidden_size=256, num_dimensions=256,
            shared_memory=None
        ))
    elif core_name == 'Lstm':
        return snt.LSTM(256)


class PixelControlTest(absltest.TestCase):

    def test_call(self):
        batch_size = 32
        feature_size = 256
        dummy_input = tf.random.normal(shape=[batch_size, feature_size], dtype=tf.float32)
        num_actions = 18
        expected_output_shape = [batch_size, 20, 20, num_actions]
        pc_net = auxiliary.PixelControl(num_actions=num_actions)
        pc_q_vals = pc_net(dummy_input)
        self.assertEqual(pc_q_vals.shape, expected_output_shape)


class RNNPixelControlNetworkTest(parameterized.TestCase):

    @parameterized.named_parameters(*TEST_CASES_RNN_PIXEL_CONTROL_NET)
    def test_call(self, embed_name: str, core_name: str):
        batch_size = 32
        observation_shape = [84, 84, 3]
        dummy_input = tf.random.uniform(shape=[batch_size] + observation_shape,
                                        minval=0.0, maxval=1.0, dtype=tf.float32)
        num_actions = 18
        expected_output_shape = [batch_size, 20, 20, num_actions]
        embed = make_embed(embed_name)
        core = make_core(core_name)
        pc_net = auxiliary.RNNPixelControlNetwork(
            embed=embed, core=core, num_actions=num_actions)
        state = pc_net.initial_state(batch_size)
        pc_q_vals = pc_net(dummy_input, state)
        self.assertEqual(pc_q_vals.shape, expected_output_shape)

    @parameterized.named_parameters(*TEST_CASES_RNN_PIXEL_CONTROL_NET)
    def test_unroll(self, embed_name: str, core_name: str):
        sequence_len = 20
        batch_size = 32
        observation_shape = [84, 84, 3]
        dummy_input_sequence = tf.random.uniform(
            shape=[sequence_len, batch_size] + observation_shape,
            minval=0.0, maxval=1.0, dtype=tf.float32)
        num_actions = 18
        expected_output_shape = [sequence_len, batch_size, 20, 20, num_actions]
        embed = make_embed(embed_name)
        core = make_core(core_name)
        pc_net = auxiliary.RNNPixelControlNetwork(
            embed=embed, core=core, num_actions=num_actions)
        state = pc_net.initial_state(batch_size)
        pc_q_vals = pc_net.unroll(dummy_input_sequence, state)
        self.assertEqual(pc_q_vals.shape, expected_output_shape)


class RewardPredictionNetworkTest(parameterized.TestCase):

    @parameterized.named_parameters(*TEST_CASES_REWARD_PREDICTION)
    def test_call(self, embed_name: str):
        sequence_len = 3
        batch_size = 32
        observation_shape = [84, 84, 3]
        dummy_input = tf.random.uniform(shape=[batch_size, sequence_len] + observation_shape,
                                        minval=0.0, maxval=1.0, dtype=tf.float32)
        expected_output_shape = [batch_size, 3]
        embed = make_embed(embed_name)
        rp_net = auxiliary.RewardPredictionNetwork(
            embed=embed, hidden_size=128)
        output = rp_net(dummy_input)
        self.assertEqual(output.shape, expected_output_shape)


if __name__ == '__main__':
    absltest.main()
