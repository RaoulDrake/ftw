from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized

from ftw.tf.networks.dnc import access
from ftw.tf.networks import recurrence as recurrence

from trfl import distribution_ops

import numpy as np
import tree
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

# TODO: implement dnc_clip_value checks
# TODO: implement tests for multiple slow cores
tfd = tfp.distributions
snt_init = snt.initializers

TEST_CASES_LSTM = [
    dict(
        testcase_name='MinimalTest',
        sequence_len=10,
        batch_size=2,
        feature_size=2,
        hidden_size=2,
        num_dimensions=2,
        dnc_clip_value=None,
        use_dnc_linear_projection=False,
        init_scale=0.3,
        min_scale=1e-6,
        tanh_mean=False,
        fixed_scale=False,
        use_tfd_independent=False,
        w_init=tf.initializers.VarianceScaling(1),
        b_init=tf.initializers.Zeros(),
        strict_period_order=True,
        period=5,
        memory_size=2,
        word_size=2,
        num_reads=4,
        num_writes=1
    ),
    dict(
        testcase_name='PeriodGreaterThanSequenceLen',
        sequence_len=2,
        batch_size=2,
        feature_size=2,
        hidden_size=2,
        num_dimensions=2,
        dnc_clip_value=None,
        use_dnc_linear_projection=False,
        init_scale=0.1,
        min_scale=1e-6,
        tanh_mean=False,
        fixed_scale=False,
        use_tfd_independent=False,
        w_init=tf.initializers.VarianceScaling(1),
        b_init=tf.initializers.Zeros(),
        strict_period_order=True,
        period=5,
        memory_size=2,
        word_size=2,
        num_reads=4,
        num_writes=1
    ),
    dict(
        testcase_name='ZeroMinScale',
        sequence_len=10,
        batch_size=2,
        feature_size=2,
        hidden_size=2,
        num_dimensions=2,
        dnc_clip_value=None,
        use_dnc_linear_projection=False,
        init_scale=0.1,
        min_scale=0.0,
        tanh_mean=False,
        fixed_scale=False,
        use_tfd_independent=False,
        w_init=tf.initializers.VarianceScaling(1),
        b_init=tf.initializers.Zeros(),
        strict_period_order=True,
        period=5,
        memory_size=2,
        word_size=2,
        num_reads=4,
        num_writes=2
    ),
    dict(
        testcase_name='FixedScale',
        sequence_len=10,
        batch_size=2,
        feature_size=2,
        hidden_size=2,
        num_dimensions=2,
        dnc_clip_value=None,
        use_dnc_linear_projection=False,
        init_scale=0.1,
        min_scale=1e-6,
        tanh_mean=False,
        fixed_scale=True,
        use_tfd_independent=False,
        w_init=tf.initializers.VarianceScaling(1),
        b_init=tf.initializers.Zeros(),
        strict_period_order=True,
        period=5,
        memory_size=2,
        word_size=2,
        num_reads=4,
        num_writes=2
    ),
    dict(
        testcase_name='TfdIndependent',
        sequence_len=10,
        batch_size=2,
        feature_size=2,
        hidden_size=2,
        num_dimensions=2,
        dnc_clip_value=None,
        use_dnc_linear_projection=False,
        init_scale=0.1,
        min_scale=1e-6,
        tanh_mean=False,
        fixed_scale=False,
        use_tfd_independent=True,
        w_init=tf.initializers.VarianceScaling(1),
        b_init=tf.initializers.Zeros(),
        strict_period_order=True,
        period=5,
        memory_size=2,
        word_size=2,
        num_reads=4,
        num_writes=2
    ),
]

TEST_CASES_DNC = TEST_CASES_LSTM + [
    dict(
        testcase_name='DncClipValue',
        sequence_len=10,
        batch_size=2,
        feature_size=2,
        hidden_size=2,
        num_dimensions=2,
        dnc_clip_value=20,
        use_dnc_linear_projection=False,
        init_scale=0.1,
        min_scale=1e-6,
        tanh_mean=False,
        fixed_scale=False,
        use_tfd_independent=False,
        w_init=tf.initializers.VarianceScaling(1),
        b_init=tf.initializers.Zeros(),
        strict_period_order=True,
        period=5,
        memory_size=2,
        word_size=2,
        num_reads=4,
        num_writes=1
    ),
    dict(
        testcase_name='DncLinearProjection',
        sequence_len=10,
        batch_size=2,
        feature_size=2,
        hidden_size=2,
        num_dimensions=2,
        dnc_clip_value=None,
        use_dnc_linear_projection=True,
        init_scale=0.1,
        min_scale=1e-6,
        tanh_mean=False,
        fixed_scale=False,
        use_tfd_independent=False,
        w_init=tf.initializers.VarianceScaling(1),
        b_init=tf.initializers.Zeros(),
        strict_period_order=True,
        period=5,
        memory_size=2,
        word_size=2,
        num_reads=4,
        num_writes=1
    ),
    dict(
        testcase_name='DncMultipleWriteHeads',
        sequence_len=10,
        batch_size=2,
        feature_size=2,
        hidden_size=2,
        num_dimensions=2,
        dnc_clip_value=None,
        use_dnc_linear_projection=False,
        init_scale=0.1,
        min_scale=1e-6,
        tanh_mean=False,
        fixed_scale=False,
        use_tfd_independent=False,
        w_init=tf.initializers.VarianceScaling(1),
        b_init=tf.initializers.Zeros(),
        strict_period_order=True,
        period=5,
        memory_size=2,
        word_size=2,
        num_reads=4,
        num_writes=2
    )
]


# class DNCWrapperTest(absltest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         super(DNCWrapperTest, cls).setUpClass()
#         ...
#
#     def setUp(self):
#         super(DNCWrapperTest, self).setUp()
#         ...
#
#     def tearDown(self):
#         super(DNCWrapperTest, self).tearDown()
#         ...
#
#     @classmethod
#     def tearDownClass(cls):
#         super(DNCWrapperTest, cls).tearDownClass()
#         ...
#
#     def test_call(self):
#         ...
#
#     def test_unroll(self):
#         ...
#
#
class VariationalUnitTest(parameterized.TestCase):

    @classmethod
    def setUpClass(cls):
        super(VariationalUnitTest, cls).setUpClass()
        ...

    def setUp(self):
        super(VariationalUnitTest, self).setUp()
        ...

    def tearDown(self):
        super(VariationalUnitTest, self).tearDown()
        ...

    @classmethod
    def tearDownClass(cls):
        super(VariationalUnitTest, cls).tearDownClass()
        ...

    @parameterized.named_parameters(*TEST_CASES_DNC)
    def test_call_dnc_core(self, sequence_len: int, batch_size: int, feature_size: int,
                           hidden_size: int, num_dimensions: int,
                           dnc_clip_value: Optional[int],
                           use_dnc_linear_projection: bool,
                           init_scale: float,
                           min_scale: float,
                           tanh_mean: bool,
                           fixed_scale: bool,
                           use_tfd_independent: bool,
                           w_init: snt_init.Initializer,
                           b_init: snt_init.Initializer,
                           strict_period_order: bool,
                           period: int,
                           memory_size: int, word_size: int, num_reads: int, num_writes: int):
        mem = access.MemoryAccess(
            memory_size=memory_size, word_size=word_size,
            num_reads=num_reads, num_writes=num_writes)
        vu = recurrence.VariationalUnit(
            hidden_size=hidden_size, num_dimensions=num_dimensions,
            dnc_clip_value=dnc_clip_value, use_dnc_linear_projection=use_dnc_linear_projection,
            init_scale=init_scale, min_scale=min_scale, tanh_mean=tanh_mean, fixed_scale=fixed_scale,
            w_init=w_init, b_init=b_init,
            shared_memory=mem)
        dummy_input = tf.ones(shape=(batch_size, feature_size), dtype=tf.float32)
        state = vu.initial_state(batch_size)
        out, state = vu(dummy_input, state)
        distribution = tfd.MultivariateNormalDiag(loc=out.loc, scale_diag=out.scale)
        self._check_distribution_shapes(distribution=distribution,
                                        expected_batch_shape=[batch_size],
                                        expected_event_shape=[num_dimensions])

    def _check_distribution_shapes(self, distribution,
                                   expected_batch_shape, expected_event_shape):
        dist_batch_shape = distribution.batch_shape.as_list()
        self.assertSequenceEqual(dist_batch_shape, expected_batch_shape)
        dist_event_shape = distribution.event_shape.as_list()
        self.assertSequenceEqual(dist_event_shape, expected_event_shape)
        z = distribution.sample()
        self.assertSequenceEqual(z.shape, expected_batch_shape + expected_event_shape)
        mean = distribution.mean()
        self.assertSequenceEqual(mean.shape, expected_batch_shape + expected_event_shape)
        stddev = distribution.stddev()
        self.assertSequenceEqual(stddev.shape, expected_batch_shape + expected_event_shape)
        variance = distribution.variance()
        self.assertSequenceEqual(variance.shape, expected_batch_shape + expected_event_shape)

    @parameterized.named_parameters(*TEST_CASES_LSTM)
    def test_call_lstm_core(self, sequence_len: int, batch_size: int, feature_size: int,
                            hidden_size: int, num_dimensions: int,
                            dnc_clip_value: Optional[int],
                            use_dnc_linear_projection: bool,
                            init_scale: float,
                            min_scale: float,
                            tanh_mean: bool,
                            fixed_scale: bool,
                            use_tfd_independent: bool,
                            w_init: snt_init.Initializer,
                            b_init: snt_init.Initializer,
                            strict_period_order: bool,
                            period: int,
                            memory_size: int, word_size: int, num_reads: int, num_writes: int):
        vu = recurrence.VariationalUnit(
            hidden_size=hidden_size, num_dimensions=num_dimensions,
            dnc_clip_value=dnc_clip_value, use_dnc_linear_projection=use_dnc_linear_projection,
            init_scale=init_scale, min_scale=min_scale, tanh_mean=tanh_mean, fixed_scale=fixed_scale,
            w_init=w_init, b_init=b_init,
            shared_memory=None)
        dummy_input = tf.ones(shape=(batch_size, feature_size), dtype=tf.float32)
        state = vu.initial_state(batch_size)
        out, state = vu(dummy_input, state)
        distribution = tfd.MultivariateNormalDiag(loc=out.loc, scale_diag=out.scale)
        self._check_distribution_shapes(distribution=distribution,
                                        expected_batch_shape=[batch_size],
                                        expected_event_shape=[num_dimensions])


class PeriodicVariationalUnitTest(parameterized.TestCase):

    @classmethod
    def setUpClass(cls):
        super(PeriodicVariationalUnitTest, cls).setUpClass()
        ...

    def setUp(self):
        super(PeriodicVariationalUnitTest, self).setUp()
        ...

    def tearDown(self):
        super(PeriodicVariationalUnitTest, self).tearDown()
        ...

    @classmethod
    def tearDownClass(cls):
        super(PeriodicVariationalUnitTest, cls).tearDownClass()
        ...

    @parameterized.named_parameters(*TEST_CASES_DNC)
    def test_call_dnc_core(self, sequence_len: int, batch_size: int, feature_size: int,
                           hidden_size: int, num_dimensions: int,
                           dnc_clip_value: Optional[int],
                           use_dnc_linear_projection: bool,
                           init_scale: float,
                           min_scale: float,
                           tanh_mean: bool,
                           fixed_scale: bool,
                           use_tfd_independent: bool,
                           w_init: snt_init.Initializer,
                           b_init: snt_init.Initializer,
                           strict_period_order: bool,
                           period: int,
                           memory_size: int, word_size: int, num_reads: int, num_writes: int):
        mem = access.MemoryAccess(
            memory_size=memory_size, word_size=word_size,
            num_reads=num_reads, num_writes=num_writes)
        periodic_vu = recurrence.PeriodicVariationalUnit(
            period=period, hidden_size=hidden_size, num_dimensions=num_dimensions,
            dnc_clip_value=dnc_clip_value, use_dnc_linear_projection=use_dnc_linear_projection,
            init_scale=init_scale, min_scale=min_scale, tanh_mean=tanh_mean, fixed_scale=fixed_scale,
            w_init=w_init, b_init=b_init,
            shared_memory=mem)
        dummy_input = tf.ones(shape=(batch_size, feature_size), dtype=tf.float32)
        state = periodic_vu.initial_state(batch_size)
        slow_loc_list = []
        slow_scale_list = []
        slow_controller_hidden_state_list = []
        slow_controller_cell_state_list = []
        slow_access_output_list = []

        # Do period - 1 calls, where the slow core's output and state should not change.
        for i in range(period):
            # Check if step counter is correct, i.e. counter == i
            i_batch_vector = np.ones(shape=[batch_size], dtype=np.float32) * i
            np.testing.assert_array_equal(i_batch_vector, state.step)

            # Do a forward pass.
            out, state = periodic_vu(dummy_input, state)
            slow_loc = out.loc
            slow_scale = out.scale

            slow_controller_hidden_state = state.core_state.controller_state.hidden
            slow_controller_cell_state = state.core_state.controller_state.cell
            slow_access_output = state.core_state.access_output

            if i > 0:
                # Check if slow core's output and state did not change.
                np.testing.assert_array_equal(slow_loc.numpy(), slow_loc_list[-1].numpy())
                np.testing.assert_array_equal(slow_scale.numpy(), slow_scale_list[-1].numpy())
                np.testing.assert_array_equal(slow_controller_hidden_state.numpy(),
                                              slow_controller_hidden_state_list[-1].numpy())
                np.testing.assert_array_equal(slow_controller_cell_state.numpy(),
                                              slow_controller_cell_state_list[-1].numpy())
                np.testing.assert_array_equal(slow_access_output.numpy(),
                                              slow_access_output_list[-1].numpy())
                # Check if step counter is correct, i.e. counter == i + 1
                i_batch_vector = np.ones(shape=[batch_size], dtype=np.float32) * (i + 1)
                np.testing.assert_array_equal(i_batch_vector, state.step)

            slow_loc_list.append(slow_loc)
            slow_scale_list.append(slow_scale)
            slow_controller_hidden_state_list.append(slow_controller_hidden_state)
            slow_controller_cell_state_list.append(slow_controller_cell_state)
            slow_access_output_list.append(slow_access_output)

        # Check if step counter is correct, i.e. counter == period
        i_batch_vector = np.ones(shape=[batch_size], dtype=np.float32) * period
        np.testing.assert_array_equal(i_batch_vector, state.step)

        # With the next call, the slow core's output and state should change.
        out, state = periodic_vu(dummy_input, state)
        slow_loc = out.loc
        slow_scale = out.scale

        slow_controller_hidden_state = state.core_state.controller_state.hidden
        slow_controller_cell_state = state.core_state.controller_state.cell
        slow_access_output = state.core_state.access_output

        # Check if slow core's output and state did change.
        # We do so by checking if assert_array_equal raises an exception, i.e. at least
        # one element in a corresponding array did change.
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal,
            slow_loc.numpy(), slow_loc_list[-1].numpy())
        if not fixed_scale:
            np.testing.assert_raises(
                AssertionError, np.testing.assert_array_equal,
                slow_scale.numpy(), slow_scale_list[-1].numpy())
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal,
            slow_controller_hidden_state.numpy(), slow_controller_hidden_state_list[-1].numpy())
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal,
            slow_controller_cell_state.numpy(), slow_controller_cell_state_list[-1].numpy())
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal,
            slow_access_output.numpy(), slow_access_output_list[-1].numpy())

        # Check if step counter is correct, i.e. counter == period + 1
        i_batch_vector = np.ones(shape=[batch_size], dtype=np.float32) * (period + 1)
        np.testing.assert_array_equal(i_batch_vector, state.step)

    @parameterized.named_parameters(*TEST_CASES_LSTM)
    def test_call_lstm_core(self, sequence_len: int, batch_size: int, feature_size: int,
                            hidden_size: int, num_dimensions: int,
                            dnc_clip_value: Optional[int],
                            use_dnc_linear_projection: bool,
                            init_scale: float,
                            min_scale: float,
                            tanh_mean: bool,
                            fixed_scale: bool,
                            use_tfd_independent: bool,
                            w_init: snt_init.Initializer,
                            b_init: snt_init.Initializer,
                            strict_period_order: bool,
                            period: int,
                            memory_size: int, word_size: int, num_reads: int, num_writes: int):
        periodic_vu = recurrence.PeriodicVariationalUnit(
            period=period, hidden_size=hidden_size, num_dimensions=num_dimensions,
            dnc_clip_value=dnc_clip_value, use_dnc_linear_projection=use_dnc_linear_projection,
            init_scale=init_scale, min_scale=min_scale, tanh_mean=tanh_mean, fixed_scale=fixed_scale,
            w_init=w_init, b_init=b_init,
            shared_memory=None)
        dummy_input = tf.ones(shape=(batch_size, feature_size), dtype=tf.float32)
        state = periodic_vu.initial_state(batch_size)
        slow_loc_list = []
        slow_scale_list = []
        slow_hidden_states_list = []
        slow_cell_states_list = []

        # Do period - 1 calls, where the slow core's output and state should not change.
        for i in range(period):
            # Check if step counter is correct, i.e. counter == i
            i_batch_vector = np.ones(shape=[batch_size], dtype=np.float32) * i
            np.testing.assert_array_equal(i_batch_vector, state.step)

            # Do a forward pass.
            out, state = periodic_vu(dummy_input, state)
            slow_loc = out.loc
            slow_scale = out.scale

            slow_hidden_state = state.core_state.hidden

            slow_cell_state = state.core_state.cell

            if i > 0:
                # Check if slow core's output and state did not change.
                np.testing.assert_array_equal(slow_loc.numpy(), slow_loc_list[-1].numpy())
                np.testing.assert_array_equal(slow_scale.numpy(), slow_scale_list[-1].numpy())
                np.testing.assert_array_equal(slow_hidden_state.numpy(),
                                              slow_hidden_states_list[-1].numpy())
                np.testing.assert_array_equal(slow_cell_state.numpy(),
                                              slow_cell_states_list[-1].numpy())
                # Check if step counter is correct, i.e. counter == i + 1
                i_batch_vector = np.ones(shape=[batch_size], dtype=np.float32) * (i + 1)
                np.testing.assert_array_equal(i_batch_vector, state.step)

            slow_loc_list.append(slow_loc)
            slow_scale_list.append(slow_scale)
            slow_hidden_states_list.append(slow_hidden_state)
            slow_cell_states_list.append(slow_cell_state)

        # Check if step counter is correct, i.e. counter == period
        i_batch_vector = np.ones(shape=[batch_size], dtype=np.float32) * period
        np.testing.assert_array_equal(i_batch_vector, state.step)

        # With the next call, the slow core's output and state should change.
        out, state = periodic_vu(dummy_input, state)
        slow_loc = out.loc
        slow_scale = out.scale

        slow_hidden_state = state.core_state.hidden
        slow_cell_state = state.core_state.cell

        # Check if slow core's output and state did change.
        # We do so by checking if assert_array_equal raises an exception, i.e. at least
        # one element in a corresponding array did change.
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal,
            slow_loc.numpy(), slow_loc_list[-1].numpy())
        if not fixed_scale:
            np.testing.assert_raises(
                AssertionError, np.testing.assert_array_equal,
                slow_scale.numpy(), slow_scale_list[-1].numpy())
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal,
            slow_hidden_state.numpy(), slow_hidden_states_list[-1].numpy())
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal,
            slow_cell_state.numpy(), slow_cell_states_list[-1].numpy())

        # Check if step counter is correct, i.e. counter == period + 1
        i_batch_vector = np.ones(shape=[batch_size], dtype=np.float32) * (period + 1)
        np.testing.assert_array_equal(i_batch_vector, state.step)


class RPTHTest(parameterized.TestCase):

    @classmethod
    def setUpClass(cls):
        super(RPTHTest, cls).setUpClass()
        ...

    def setUp(self):
        super(RPTHTest, self).setUp()
        ...

    def tearDown(self):
        super(RPTHTest, self).tearDown()
        ...

    @classmethod
    def tearDownClass(cls):
        super(RPTHTest, cls).tearDownClass()
        ...

    @parameterized.named_parameters(*TEST_CASES_DNC)
    def test_call_dnc_cores(self, sequence_len: int, batch_size: int, feature_size: int,
                            hidden_size: int, num_dimensions: int,
                            dnc_clip_value: Optional[int],
                            use_dnc_linear_projection: bool,
                            init_scale: float,
                            min_scale: float,
                            tanh_mean: bool,
                            fixed_scale: bool,
                            use_tfd_independent: bool,
                            w_init: snt_init.Initializer,
                            b_init: snt_init.Initializer,
                            strict_period_order: bool,
                            period: int,
                            memory_size: int, word_size: int, num_reads: int, num_writes: int):
        """This is essentially a sanity check that does the same test as in the corresponding test
        in PeriodicVariationalUnitTest. Thus, we only check if nothing in RPTH modifies the expected
        behaviour of PeriodicVariationalUnit.
        """
        mem = access.MemoryAccess(
            memory_size=memory_size, word_size=word_size,
            num_reads=num_reads, num_writes=num_writes)
        rpth = recurrence.RPTH(
            period=period, hidden_size=hidden_size, num_dimensions=num_dimensions,
            dnc_clip_value=dnc_clip_value, use_dnc_linear_projection=use_dnc_linear_projection,
            init_scale=init_scale, min_scale=min_scale, tanh_mean=tanh_mean, fixed_scale=fixed_scale,
            use_tfd_independent=use_tfd_independent, w_init=w_init, b_init=b_init,
            strict_period_order=strict_period_order,
            shared_memory=mem)
        dummy_input = tf.ones(shape=(batch_size, feature_size), dtype=tf.float32)
        state = rpth.initial_state(batch_size)
        slow_loc_list = []
        slow_scale_list = []
        slow_controller_hidden_state_list = []
        slow_controller_cell_state_list = []
        slow_access_output_list = []

        # Do period - 1 calls, where the slow core's output and state should not change.
        for i in range(period):
            # Check if step counter is correct, i.e. counter == i
            i_batch_vector = np.ones(shape=[batch_size], dtype=np.float32) * i
            np.testing.assert_array_equal(i_batch_vector, state.step)

            # Do a forward pass.
            out, state = rpth(dummy_input, state)
            distribution_params = out.distribution_params
            slow_params = distribution_params[0]
            slow_loc = slow_params.loc
            slow_scale = slow_params.scale

            slow_controller_hidden_state = state.core_state.controller_state[0].hidden
            slow_controller_cell_state = state.core_state.controller_state[0].cell
            slow_access_output = state.core_state.access_output[0]

            if i > 0:
                # Check if slow core's output and state did not change.
                np.testing.assert_array_equal(slow_loc.numpy(), slow_loc_list[-1].numpy())
                np.testing.assert_array_equal(slow_scale.numpy(), slow_scale_list[-1].numpy())
                np.testing.assert_array_equal(slow_controller_hidden_state.numpy(),
                                              slow_controller_hidden_state_list[-1].numpy())
                np.testing.assert_array_equal(slow_controller_cell_state.numpy(),
                                              slow_controller_cell_state_list[-1].numpy())
                np.testing.assert_array_equal(slow_access_output.numpy(),
                                              slow_access_output_list[-1].numpy())
                # Check if step counter is correct, i.e. counter == i + 1
                i_batch_vector = np.ones(shape=[batch_size], dtype=np.float32) * (i + 1)
                np.testing.assert_array_equal(i_batch_vector, state.step)

            slow_loc_list.append(slow_loc)
            slow_scale_list.append(slow_scale)
            slow_controller_hidden_state_list.append(slow_controller_hidden_state)
            slow_controller_cell_state_list.append(slow_controller_cell_state)
            slow_access_output_list.append(slow_access_output)

        # Check if step counter is correct, i.e. counter == period
        i_batch_vector = np.ones(shape=[batch_size], dtype=np.float32) * period
        np.testing.assert_array_equal(i_batch_vector, state.step)

        # With the next call, the slow core's output and state should change.
        out, state = rpth(dummy_input, state)
        distribution_params = out.distribution_params
        slow_params = distribution_params[0]
        slow_loc = slow_params.loc
        slow_scale = slow_params.scale

        slow_controller_hidden_state = state.core_state.controller_state[0].hidden
        slow_controller_cell_state = state.core_state.controller_state[0].cell
        slow_access_output = state.core_state.access_output[0]

        # Check if slow core's output and state did change.
        # We do so by checking if assert_array_equal raises an exception, i.e. at least
        # one element in a corresponding array did change.
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal,
            slow_loc.numpy(), slow_loc_list[-1].numpy())
        if not fixed_scale:
            np.testing.assert_raises(
                AssertionError, np.testing.assert_array_equal,
                slow_scale.numpy(), slow_scale_list[-1].numpy())
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal,
            slow_controller_hidden_state.numpy(), slow_controller_hidden_state_list[-1].numpy())
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal,
            slow_controller_cell_state.numpy(), slow_controller_cell_state_list[-1].numpy())
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal,
            slow_access_output.numpy(), slow_access_output_list[-1].numpy())

        # Check if step counter is correct, i.e. counter == period + 1
        i_batch_vector = np.ones(shape=[batch_size], dtype=np.float32) * (period + 1)
        np.testing.assert_array_equal(i_batch_vector, state.step)

    @parameterized.named_parameters(*TEST_CASES_LSTM)
    def test_call_lstm_cores(self, sequence_len: int, batch_size: int, feature_size: int,
                             hidden_size: int, num_dimensions: int,
                             dnc_clip_value: Optional[int],
                             use_dnc_linear_projection: bool,
                             init_scale: float,
                             min_scale: float,
                             tanh_mean: bool,
                             fixed_scale: bool,
                             use_tfd_independent: bool,
                             w_init: snt_init.Initializer,
                             b_init: snt_init.Initializer,
                             strict_period_order: bool,
                             period: int,
                             memory_size: int, word_size: int, num_reads: int, num_writes: int):
        """This is essentially a sanity check that does the same test as in the corresponding test
        in PeriodicVariationalUnitTest. Thus, we only check if nothing in RPTH modifies the expected
        behaviour of PeriodicVariationalUnit.
        """
        rpth = recurrence.RPTH(
            period=period, hidden_size=hidden_size, num_dimensions=num_dimensions,
            dnc_clip_value=dnc_clip_value, use_dnc_linear_projection=use_dnc_linear_projection,
            init_scale=init_scale, min_scale=min_scale, tanh_mean=tanh_mean, fixed_scale=fixed_scale,
            use_tfd_independent=use_tfd_independent, w_init=w_init, b_init=b_init,
            strict_period_order=strict_period_order,
            shared_memory=None)
        dummy_input = tf.ones(shape=(batch_size, feature_size), dtype=tf.float32)
        state = rpth.initial_state(batch_size)
        slow_loc_list = []
        slow_scale_list = []
        slow_hidden_states_list = []
        slow_cell_states_list = []

        # Do period - 1 calls, where the slow core's output and state should not change.
        for i in range(period):
            # Check if step counter is correct, i.e. counter == i
            i_batch_vector = np.ones(shape=[batch_size], dtype=np.float32) * i
            np.testing.assert_array_equal(i_batch_vector, state.step)

            # Do a forward pass.
            out, state = rpth(dummy_input, state)
            distribution_params = out.distribution_params
            slow_params = distribution_params[0]
            slow_loc = slow_params.loc
            slow_scale = slow_params.scale

            slow_hidden_state = state.core_state.hidden[0]

            slow_cell_state = state.core_state.cell[0]

            if i > 0:
                # Check if slow core's output and state did not change.
                np.testing.assert_array_equal(slow_loc.numpy(), slow_loc_list[-1].numpy())
                np.testing.assert_array_equal(slow_scale.numpy(), slow_scale_list[-1].numpy())
                np.testing.assert_array_equal(slow_hidden_state.numpy(),
                                              slow_hidden_states_list[-1].numpy())
                np.testing.assert_array_equal(slow_cell_state.numpy(),
                                              slow_cell_states_list[-1].numpy())
                # Check if step counter is correct, i.e. counter == i + 1
                i_batch_vector = np.ones(shape=[batch_size], dtype=np.float32) * (i + 1)
                np.testing.assert_array_equal(i_batch_vector, state.step)

            slow_loc_list.append(slow_loc)
            slow_scale_list.append(slow_scale)
            slow_hidden_states_list.append(slow_hidden_state)
            slow_cell_states_list.append(slow_cell_state)

        # Check if step counter is correct, i.e. counter == period
        i_batch_vector = np.ones(shape=[batch_size], dtype=np.float32) * period
        np.testing.assert_array_equal(i_batch_vector, state.step)

        # With the next call, the slow core's output and state should change.
        out, state = rpth(dummy_input, state)
        distribution_params = out.distribution_params
        slow_params = distribution_params[0]
        slow_loc = slow_params.loc
        slow_scale = slow_params.scale
        slow_hidden_state = state.core_state.hidden[0]
        slow_cell_state = state.core_state.cell[0]

        # Check if slow core's output and state did change.
        # We do so by checking if assert_array_equal raises an exception, i.e. at least
        # one element in a corresponding array did change.
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal,
            slow_loc.numpy(), slow_loc_list[-1].numpy())
        if not fixed_scale:
            np.testing.assert_raises(
                AssertionError, np.testing.assert_array_equal,
                slow_scale.numpy(), slow_scale_list[-1].numpy())
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal,
            slow_hidden_state.numpy(), slow_hidden_states_list[-1].numpy())
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal,
            slow_cell_state.numpy(), slow_cell_states_list[-1].numpy())

        # Check if step counter is correct, i.e. counter == period + 1
        i_batch_vector = np.ones(shape=[batch_size], dtype=np.float32) * (period + 1)
        np.testing.assert_array_equal(i_batch_vector, state.step)

    @parameterized.named_parameters(*TEST_CASES_DNC)
    def test_unroll_dnc_cores(self, sequence_len: int, batch_size: int, feature_size: int,
                              hidden_size: int, num_dimensions: int,
                              dnc_clip_value: Optional[int],
                              use_dnc_linear_projection: bool,
                              init_scale: float,
                              min_scale: float,
                              tanh_mean: bool,
                              fixed_scale: bool,
                              use_tfd_independent: bool,
                              w_init: snt_init.Initializer,
                              b_init: snt_init.Initializer,
                              strict_period_order: bool,
                              period: int,
                              memory_size: int, word_size: int, num_reads: int, num_writes: int):
        mem = access.MemoryAccess(
            memory_size=memory_size, word_size=word_size,
            num_reads=num_reads, num_writes=num_writes)
        rpth = recurrence.RPTH(
            period=period, hidden_size=hidden_size, num_dimensions=num_dimensions,
            dnc_clip_value=dnc_clip_value, use_dnc_linear_projection=use_dnc_linear_projection,
            init_scale=init_scale, min_scale=min_scale, tanh_mean=tanh_mean, fixed_scale=fixed_scale,
            use_tfd_independent=use_tfd_independent, w_init=w_init, b_init=b_init,
            strict_period_order=strict_period_order,
            shared_memory=mem)

        # Static unroll test:
        self._unroll_checks(
            core=rpth, sequence_len=sequence_len, batch_size=batch_size, feature_size=feature_size,
            period=period)

    @parameterized.named_parameters(*TEST_CASES_LSTM)
    def test_unroll_lstm_cores(self, sequence_len: int, batch_size: int, feature_size: int,
                               hidden_size: int, num_dimensions: int,
                               dnc_clip_value: Optional[int],
                               use_dnc_linear_projection: bool,
                               init_scale: float,
                               min_scale: float,
                               tanh_mean: bool,
                               fixed_scale: bool,
                               use_tfd_independent: bool,
                               w_init: snt_init.Initializer,
                               b_init: snt_init.Initializer,
                               strict_period_order: bool,
                               period: int,
                               memory_size: int, word_size: int, num_reads: int, num_writes: int):
        rpth = recurrence.RPTH(
            period=period, hidden_size=hidden_size, num_dimensions=num_dimensions,
            dnc_clip_value=dnc_clip_value, use_dnc_linear_projection=use_dnc_linear_projection,
            init_scale=init_scale, min_scale=min_scale, tanh_mean=tanh_mean, fixed_scale=fixed_scale,
            use_tfd_independent=use_tfd_independent, w_init=w_init, b_init=b_init,
            strict_period_order=strict_period_order,
            shared_memory=None)

        # Static unroll test:
        self._unroll_checks(
            core=rpth, sequence_len=sequence_len, batch_size=batch_size, feature_size=feature_size,
            period=period)

    def _unroll_checks(self, core, sequence_len, batch_size, feature_size, period):
        unroll_state = core.initial_state(batch_size)

        unroll_out, unroll_state, grads = self._unroll_and_get_gradients(
            core=core, state=unroll_state,
            sequence_len=sequence_len, batch_size=batch_size, feature_size=feature_size)

        # Check shapes of output and state.
        self._check_shapes(nested_tensor=unroll_out, expected_shape=[sequence_len, batch_size, feature_size])
        self._check_shapes(nested_tensor=unroll_state, expected_shape=[batch_size])

        # Check that we have no zero/None/NaN gradients, i.e. there is no zero/None/NaN in any gradient.
        # In case period > sequence_len, there will be zero gradients for the slow core, since it never participated
        # in the forward pass. Thus, we do not check gradients in that case.
        if not period > sequence_len:
            self._check_gradients(gradients=grads)

    def _unroll_and_get_gradients(self, core, state,
                                  sequence_len, batch_size, feature_size):
        input_sequence = tf.ones(shape=(sequence_len, batch_size, feature_size),
                                       dtype=tf.float32)
        with tf.GradientTape() as tape:
            unroll_out, unroll_state = snt.static_unroll(
                core=core, input_sequence=input_sequence, initial_state=state,
                sequence_length=sequence_len)
            dist_params = unroll_out.distribution_params
            prior_params = dist_params[0]
            posterior_params = dist_params[1]
            kld_mean, kld_cov = distribution_ops.factorised_kl_gaussian(
                dist1_mean=posterior_params.loc,
                dist1_covariance_or_scale=posterior_params.scale,
                dist2_mean=prior_params.loc,
                dist2_covariance_or_scale=prior_params.scale,
                both_diagonal=True)
            kld_loss = 1e-3 * tf.reduce_mean(kld_mean + kld_cov)
        grads = tape.gradient(kld_loss, core.trainable_variables)

        return unroll_out, unroll_state, grads

    def _check_shapes(self, nested_tensor, expected_shape):
        shapes = tree.map_structure(
            lambda t: (t.shape
                       if not isinstance(t, tfd.Distribution)
                       else t.batch_shape.as_list() + t.event_shape.as_list()),
            nested_tensor)
        if len(expected_shape) != 1:
            tree.map_structure(
                lambda x: np.testing.assert_array_equal(
                    x[:len(expected_shape)], np.array(expected_shape)),
                shapes)
        else:
            tree.map_structure(
                lambda x: self.assertEqual(x[0], expected_shape[0]),
                shapes)

    def _check_gradients(self, gradients):
        tree.map_structure(lambda x: self.assertFalse((np.nan in x.numpy())), gradients)
        tree.map_structure(lambda x: self.assertFalse((None in x.numpy())), gradients)
        tree.map_structure(lambda x: self.assertFalse((x is None)), gradients)
        tree.map_structure(lambda x: self.assertFalse((x.numpy() is None)), gradients)


if __name__ == '__main__':
    absltest.main()
