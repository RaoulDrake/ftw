from typing import Optional, Mapping, Sequence, Tuple, Any

from absl.testing import absltest
from absl.testing import parameterized

from acme import specs
from acme.adders.reverb import base

import dm_env
import numpy as np
import tree
import tensorflow as tf

from ftw.adders.reverb import multi_sequence
from ftw.adders.reverb import test_utils

TEST_CASES = [
    dict(
        testcase_name='SequencePeriodOne',
        sequence_lengths={base.DEFAULT_PRIORITY_TABLE: 3},
        periods={base.DEFAULT_PRIORITY_TABLE: 1},
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.transition(reward=3.0, observation=3)),
            (0, dm_env.transition(reward=5.0, observation=4)),
            (0, dm_env.termination(reward=7.0, observation=5)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, extra)
            [(1, 0, 2.0, 1.0, True, ()), (2, 0, 3.0, 1.0, False, ()), (3, 0, 5.0, 1.0, False, ())],
            [(2, 0, 3.0, 1.0, False, ()), (3, 0, 5.0, 1.0, False, ()), (4, 0, 7.0, 0.0, False, ())],
            [(3, 0, 5.0, 1.0, False, ()), (4, 0, 7.0, 0.0, False, ()), (5, 0, 0.0, 0.0, False, ())],
        ),
        priority_fns={base.DEFAULT_PRIORITY_TABLE: lambda x: 1.0},
        should_insert_fns={base.DEFAULT_PRIORITY_TABLE: lambda x: True}
    ),
    dict(
        testcase_name='SequencePeriodTwo',
        sequence_lengths={base.DEFAULT_PRIORITY_TABLE: 3},
        periods={base.DEFAULT_PRIORITY_TABLE: 2},
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.transition(reward=3.0, observation=3)),
            (0, dm_env.transition(reward=5.0, observation=4)),
            (0, dm_env.termination(reward=7.0, observation=5)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, extra)
            [(1, 0, 2.0, 1.0, True, ()), (2, 0, 3.0, 1.0, False, ()), (3, 0, 5.0, 1.0, False, ())],
            [(3, 0, 5.0, 1.0, False, ()), (4, 0, 7.0, 0.0, False, ()), (5, 0, 0.0, 0.0, False, ())],
        ),
        priority_fns={base.DEFAULT_PRIORITY_TABLE: lambda x: 1.0},
        should_insert_fns={base.DEFAULT_PRIORITY_TABLE: lambda x: True}
    ),
    dict(
        testcase_name='SequencePaddingPeriodTwo',
        sequence_lengths={base.DEFAULT_PRIORITY_TABLE: 4},
        periods={base.DEFAULT_PRIORITY_TABLE: 2},
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.transition(reward=3.0, observation=3)),
            (0, dm_env.transition(reward=5.0, observation=4)),
            (0, dm_env.termination(reward=7.0, observation=5)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, extra)
            [(1, 0, 2.0, 1.0, True, ()), (2, 0, 3.0, 1.0, False, ()),
             (3, 0, 5.0, 1.0, False, ()), (4, 0, 7.0, 0.0, False, ())],
            [(3, 0, 5.0, 1.0, False, ()), (4, 0, 7.0, 0.0, False, ()),
             (5, 0, 0.0, 0.0, False, ()), (0, 0, 0.0, 0.0, False, ())],
        ),
        priority_fns={base.DEFAULT_PRIORITY_TABLE: lambda x: 1.0},
        should_insert_fns={base.DEFAULT_PRIORITY_TABLE: lambda x: True}
    ),
    dict(
        testcase_name='SequencePaddingPeriodThree',
        sequence_lengths={base.DEFAULT_PRIORITY_TABLE: 4},
        periods={base.DEFAULT_PRIORITY_TABLE: 3},
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.transition(reward=3.0, observation=3)),
            (0, dm_env.transition(reward=5.0, observation=4)),
            (0, dm_env.transition(reward=7.0, observation=5)),
            (0, dm_env.transition(reward=9.0, observation=6)),
            (0, dm_env.transition(reward=11.0, observation=7)),
            (0, dm_env.termination(reward=13.0, observation=8)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode, extra)
            [(1, 0, 2.0, 1.0, True, ()), (2, 0, 3.0, 1.0, False, ()),
             (3, 0, 5.0, 1.0, False, ()), (4, 0, 7.0, 1.0, False, ())],
            [(4, 0, 7.0, 1.0, False, ()), (5, 0, 9.0, 1.0, False, ()),
             (6, 0, 11.0, 1.0, False, ()), (7, 0, 13.0, 0.0, False, ())],
            [(7, 0, 13.0, 0.0, False, ()), (8, 0, 0.0, 0.0, False, ()),
             (0, 0, 0.0, 0.0, False, ()), (0, 0, 0.0, 0.0, False, ())],
        ),
        priority_fns={base.DEFAULT_PRIORITY_TABLE: lambda x: 1.0},
        should_insert_fns={base.DEFAULT_PRIORITY_TABLE: lambda x: True}
    ),
    dict(
        testcase_name='SequenceEarlyTerminationPeriodOne',
        sequence_lengths={base.DEFAULT_PRIORITY_TABLE: 3},
        periods={base.DEFAULT_PRIORITY_TABLE: 1},
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.termination(reward=3.0, observation=3)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode, extra)
            [(1, 0, 2.0, 1.0, True, ()), (2, 0, 3.0, 0.0, False, ()),
             (3, 0, 0.0, 0.0, False, ())],),
        priority_fns={base.DEFAULT_PRIORITY_TABLE: lambda x: 1.0},
        should_insert_fns={base.DEFAULT_PRIORITY_TABLE: lambda x: True}
    ),
    dict(
        testcase_name='SequenceEarlyTerminationPeriodTwo',
        sequence_lengths={base.DEFAULT_PRIORITY_TABLE: 3},
        periods={base.DEFAULT_PRIORITY_TABLE: 2},
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.termination(reward=3.0, observation=3)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode, extra)
            [(1, 0, 2.0, 1.0, True, ()), (2, 0, 3.0, 0.0, False, ()),
             (3, 0, 0.0, 0.0, False, ())],),
        priority_fns={base.DEFAULT_PRIORITY_TABLE: lambda x: 1.0},
        should_insert_fns={base.DEFAULT_PRIORITY_TABLE: lambda x: True}
    ),
    dict(
        testcase_name='SequenceEarlyTerminationPaddingPeriodOne',
        sequence_lengths={base.DEFAULT_PRIORITY_TABLE: 4},
        periods={base.DEFAULT_PRIORITY_TABLE: 1},
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.termination(reward=3.0, observation=3)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode, extra)
            [
                (1, 0, 2.0, 1.0, True, ()),
                (2, 0, 3.0, 0.0, False, ()),
                (3, 0, 0.0, 0.0, False, ()),
                (0, 0, 0.0, 0.0, False, ()),
            ],),
        priority_fns={base.DEFAULT_PRIORITY_TABLE: lambda x: 1.0},
        should_insert_fns={base.DEFAULT_PRIORITY_TABLE: lambda x: True}
    ),
    dict(
        testcase_name='SequenceEarlyTerminationPaddingPeriodTwo',
        sequence_lengths={base.DEFAULT_PRIORITY_TABLE: 4},
        periods={base.DEFAULT_PRIORITY_TABLE: 2},
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.termination(reward=3.0, observation=3)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode, extra)
            [
                (1, 0, 2.0, 1.0, True, ()),
                (2, 0, 3.0, 0.0, False, ()),
                (3, 0, 0.0, 0.0, False, ()),
                (0, 0, 0.0, 0.0, False, ()),
            ],),
        priority_fns={base.DEFAULT_PRIORITY_TABLE: lambda x: 1.0},
        should_insert_fns={base.DEFAULT_PRIORITY_TABLE: lambda x: True}
    ),
    dict(
        testcase_name='SequenceEarlyTerminationNoPadding',
        sequence_lengths={base.DEFAULT_PRIORITY_TABLE: 4},
        periods={base.DEFAULT_PRIORITY_TABLE: 1},
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.termination(reward=3.0, observation=3)),
        ),
        expected_sequences=(
            # None, since
            # a) we do not pad and
            # b) not enough steps were appended to the writer, so writer.create_item()
            #    in SequenceAdder._maybe_add_priorities() never gets called.
            ),
        priority_fns={base.DEFAULT_PRIORITY_TABLE: lambda x: 1.0},
        should_insert_fns={base.DEFAULT_PRIORITY_TABLE: lambda x: True},
        pad_end_of_episode=False,
    ),
]

TEST_CASES_MULTI_SEQ = [
    dict(
        testcase_name='RewardPredictionTest',
        sequence_lengths={'nonzero_reward_buffer': 3,
                          'zero_reward_buffer': 3},
        periods={'nonzero_reward_buffer': 1,
                 'zero_reward_buffer': 1},
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.transition(reward=3.0, observation=3)),
            (0, dm_env.transition(reward=5.0, observation=4)),
            (0, dm_env.transition(reward=0.0, observation=5)),
            (0, dm_env.transition(reward=-7.0, observation=6)),
            (0, dm_env.transition(reward=0.0, observation=7)),
            (0, dm_env.termination(reward=-9.0, observation=8)),
        ),
        expected_sequences={
            'nonzero_reward_buffer': (
                # (observation, action, reward, discount, start_of_episode, extra)
                [(1, 0, 2.0, 1.0, True, ()), (2, 0, 3.0, 1.0, False, ()), (3, 0, 5.0, 1.0, False, ())],
                [(3, 0, 5.0, 1.0, False, ()), (4, 0, 0.0, 1.0, False, ()), (5, 0, -7.0, 1.0, False, ())],
                [(5, 0, -7.0, 1.0, False, ()), (6, 0, 0.0, 1.0, False, ()), (7, 0, -9.0, 0.0, False, ())],
            ),
            'zero_reward_buffer': (
                # (observation, action, reward, discount, start_of_episode, extra)
                [(2, 0, 3.0, 1.0, False, ()), (3, 0, 5.0, 1.0, False, ()), (4, 0, 0.0, 1.0, False, ())],
                [(4, 0, 0.0, 1.0, False, ()), (5, 0, -7.0, 1.0, False, ()), (6, 0, 0.0, 1.0, False, ())],
                [(6, 0, 0.0, 1.0, False, ()), (7, 0, -9.0, 0.0, False, ()), (8, 0, 0.0, 0.0, False, ())],
            ),
        },
        priority_fns={'nonzero_reward_buffer': lambda x: 1.0,
                      'zero_reward_buffer': lambda x: 1.0},
        should_insert_fns={'nonzero_reward_buffer': lambda x: x[-1].reward != 0.,
                           'zero_reward_buffer': lambda x: x[-1].reward == 0.}
    ),
    dict(
        testcase_name='MixedSeqLenPeriodsTest',
        sequence_lengths={'buffer_one': 3,
                          'buffer_two': 4},
        periods={'buffer_one': 2,
                 'buffer_two': 1},
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.transition(reward=3.0, observation=3)),
            (0, dm_env.transition(reward=5.0, observation=4)),
            (0, dm_env.transition(reward=0.0, observation=5)),
            (0, dm_env.transition(reward=-7.0, observation=6)),
            (0, dm_env.transition(reward=0.0, observation=7)),
            (0, dm_env.termination(reward=-9.0, observation=8)),
        ),
        expected_sequences={
            'buffer_one': (
                # (observation, action, reward, discount, start_of_episode, extra)
                [(1, 0, 2.0, 1.0, True, ()), (2, 0, 3.0, 1.0, False, ()), (3, 0, 5.0, 1.0, False, ())],
                [(3, 0, 5.0, 1.0, False, ()), (4, 0, 0.0, 1.0, False, ()), (5, 0, -7.0, 1.0, False, ())],
                [(5, 0, -7.0, 1.0, False, ()), (6, 0, 0.0, 1.0, False, ()), (7, 0, -9.0, 0.0, False, ())],
                [(7, 0, -9.0, 0.0, False, ()), (8, 0, 0.0, 0.0, False, ()), (0, 0, 0.0, 0.0, False, ())],
            ),
            'buffer_two': (
                # (observation, action, reward, discount, start_of_episode, extra)
                [(1, 0, 2.0, 1.0, True, ()), (2, 0, 3.0, 1.0, False, ()),
                 (3, 0, 5.0, 1.0, False, ()), (4, 0, 0.0, 1.0, False, ())],
                [(2, 0, 3.0, 1.0, False, ()), (3, 0, 5.0, 1.0, False, ()),
                 (4, 0, 0.0, 1.0, False, ()), (5, 0, -7.0, 1.0, False, ())],
                [(3, 0, 5.0, 1.0, False, ()), (4, 0, 0.0, 1.0, False, ()),
                 (5, 0, -7.0, 1.0, False, ()), (6, 0, 0.0, 1.0, False, ())],
                [(4, 0, 0.0, 1.0, False, ()), (5, 0, -7.0, 1.0, False, ()),
                 (6, 0, 0.0, 1.0, False, ()), (7, 0, -9.0, 0.0, False, ())],
                [(5, 0, -7.0, 1.0, False, ()), (6, 0, 0.0, 1.0, False, ()),
                 (7, 0, -9.0, 0.0, False, ()), (8, 0, 0.0, 0.0, False, ())]
            ),
        },
        priority_fns={'buffer_one': lambda x: 1.0,
                      'buffer_two': lambda x: 1.0},
        should_insert_fns={'buffer_one': lambda x: True,
                           'buffer_two': lambda x: True}
    ),
    dict(
        testcase_name='MixedSeqLenPeriodsNoPaddingTest',
        sequence_lengths={'buffer_one': 3,
                          'buffer_two': 4},
        periods={'buffer_one': 2,
                 'buffer_two': 1},
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.transition(reward=3.0, observation=3)),
            (0, dm_env.transition(reward=5.0, observation=4)),
            (0, dm_env.transition(reward=0.0, observation=5)),
            (0, dm_env.transition(reward=-7.0, observation=6)),
            (0, dm_env.transition(reward=0.0, observation=7)),
            (0, dm_env.termination(reward=-9.0, observation=8)),
        ),
        expected_sequences={
            'buffer_one': (
                # (observation, action, reward, discount, start_of_episode, extra)
                [(1, 0, 2.0, 1.0, True, ()), (2, 0, 3.0, 1.0, False, ()), (3, 0, 5.0, 1.0, False, ())],
                [(3, 0, 5.0, 1.0, False, ()), (4, 0, 0.0, 1.0, False, ()), (5, 0, -7.0, 1.0, False, ())],
                [(5, 0, -7.0, 1.0, False, ()), (6, 0, 0.0, 1.0, False, ()), (7, 0, -9.0, 0.0, False, ())]
            ),
            'buffer_two': (
                # (observation, action, reward, discount, start_of_episode, extra)
                [(1, 0, 2.0, 1.0, True, ()), (2, 0, 3.0, 1.0, False, ()),
                 (3, 0, 5.0, 1.0, False, ()), (4, 0, 0.0, 1.0, False, ())],
                [(2, 0, 3.0, 1.0, False, ()), (3, 0, 5.0, 1.0, False, ()),
                 (4, 0, 0.0, 1.0, False, ()), (5, 0, -7.0, 1.0, False, ())],
                [(3, 0, 5.0, 1.0, False, ()), (4, 0, 0.0, 1.0, False, ()),
                 (5, 0, -7.0, 1.0, False, ()), (6, 0, 0.0, 1.0, False, ())],
                [(4, 0, 0.0, 1.0, False, ()), (5, 0, -7.0, 1.0, False, ()),
                 (6, 0, 0.0, 1.0, False, ()), (7, 0, -9.0, 0.0, False, ())],
                [(5, 0, -7.0, 1.0, False, ()), (6, 0, 0.0, 1.0, False, ()),
                 (7, 0, -9.0, 0.0, False, ()), (8, 0, 0.0, 0.0, False, ())]
            ),
        },
        priority_fns={'buffer_one': lambda x: 1.0,
                      'buffer_two': lambda x: 1.0},
        should_insert_fns={'buffer_one': lambda x: True,
                           'buffer_two': lambda x: True},
        pad_end_of_episode=False
    ),
]


class MultiSequenceAdderTest(test_utils.AdderTestMixin, parameterized.TestCase):

    @parameterized.named_parameters(*TEST_CASES)
    def test_adder_sequence(self, sequence_lengths: Mapping[str, int],
                            periods: Mapping[str, int],
                            first, steps, expected_sequences,
                            priority_fns: Optional[base.PriorityFnMapping] = None,
                            should_insert_fns: Optional[multi_sequence.ShouldInsertFnMapping] = None,
                            pad_end_of_episode: bool = True):
        adder = multi_sequence.MultiSequenceAdder(
            self.client,
            sequence_lengths=sequence_lengths,
            periods=periods,
            priority_fns=priority_fns,
            should_insert_fns=should_insert_fns,
            pad_end_of_episode=pad_end_of_episode)
        super().run_test_adder(
            adder=adder,
            first=first,
            steps=steps,
            expected_items=expected_sequences)

    @parameterized.named_parameters(*TEST_CASES_MULTI_SEQ)
    def test_adder_multi_sequence(self, sequence_lengths: Mapping[str, int],
                                  periods: Mapping[str, int],
                                  first, steps, expected_sequences,
                                  priority_fns: Optional[base.PriorityFnMapping] = None,
                                  should_insert_fns: Optional[multi_sequence.ShouldInsertFnMapping] = None,
                                  pad_end_of_episode: bool = True):
        adder = multi_sequence.MultiSequenceAdder(
            self.client,
            sequence_lengths=sequence_lengths,
            periods=periods,
            priority_fns=priority_fns,
            should_insert_fns=should_insert_fns,
            pad_end_of_episode=pad_end_of_episode)
        self.run_test_adder(
            adder=adder,
            first=first,
            steps=steps,
            expected_items=expected_sequences)

    def run_test_adder(self,
                       adder: base.ReverbAdder,
                       first: dm_env.TimeStep,
                       steps: Sequence[Tuple[Any, dm_env.TimeStep]],
                       expected_items: Mapping[str, Sequence[Any]]):
        """Runs a unit test case for the adder.

        Args:
          adder: The instance of `base.ReverbAdder` that is being tested.
          first: The first `dm_env.TimeStep` that is used to call
            `base.ReverbAdder.add_first()`.
          steps: A sequence of (action, timestep) tuples that are passed to
            `base.ReverbAdder.add()`.
          expected_items: The sequence of items that are expected to be created
            by calling the adder's `add_first()` method on `first` and `add()` on
            all of the elements in `steps`.
        """
        if not steps:
            raise ValueError('At least one step must be given.')

        env_spec = tree.map_structure(
            test_utils._numeric_to_spec,
            specs.EnvironmentSpec(
                observations=steps[0][1].observation,
                actions=steps[0][0],
                rewards=steps[0][1].reward,
                discounts=steps[0][1].discount))
        signature = adder.signature(env_spec)

        # Add all the data up to the final step.
        adder.add_first(first)
        for action, ts in steps[:-1]:
            adder.add(action, next_timestep=ts)

        if len(steps) == 1:
            # adder.add() has not been called yet, so no writers have been created.
            self.assertEmpty(self.client.writers)
        else:
            # Make sure the writer has been created but not closed.
            self.assertLen(self.client.writers, 1)
            self.assertFalse(self.client.writers[0].closed)

        # Add the final step.
        adder.add(*steps[-1])

        # Ending the episode should close the writer. No new writer should yet have
        # been created as it is constructed lazily.
        self.assertLen(self.client.writers, 1)
        self.assertTrue(self.client.writers[0].closed)

        # Make sure table names of our expected and observed data match.
        expected_items_table_names = expected_items.keys()
        observed_items_table_names = [p[0] for p in self.client.writers[0].priorities]
        self.assertEqual(set(expected_items_table_names), set(observed_items_table_names))
        # Make sure our expected and observed data match.
        for table_name in expected_items_table_names:
            table_specific_observed_items = [
                p[1] for p in self.client.writers[0].priorities if p[0] == table_name]
            table_specific_expected_items = expected_items[table_name]
            for expected_item, observed_item in zip(
                    table_specific_expected_items, table_specific_observed_items):
                # Set check_types=False because
                tree.map_structure(
                    np.testing.assert_array_almost_equal,
                    expected_item,
                    observed_item,
                    check_types=False)

        def _check_signature(spec: tf.TensorSpec, value):
            # Convert int/float to numpy arrays of dtype np.int64 and np.float64.
            value = np.asarray(value)
            self.assertTrue(spec.is_compatible_with(tf.convert_to_tensor(value)))

        for step in self.client.writers[0].timesteps:
            tree.map_structure(_check_signature, signature, step)

        # Add the start of a second trajectory.
        adder.add_first(first)
        adder.add(*steps[0])

        # Make sure this creates an new writer.
        self.assertLen(self.client.writers, 2)
        # The writer is closed if the recently added `dm_env.TimeStep`'s' step_type
        # is `dm_env.StepType.LAST`.
        if steps[0][1].last():
            self.assertTrue(self.client.writers[1].closed)
        else:
            self.assertFalse(self.client.writers[1].closed)


if __name__ == '__main__':
    absltest.main()
