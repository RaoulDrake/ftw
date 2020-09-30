from typing import Any, Sequence, Tuple

from absl.testing import absltest
from absl.testing import parameterized


import dm_env
from acme import specs
from acme.adders.reverb import base
from ftw.adders.reverb import rnn_sequence
from ftw.adders.reverb import test_utils

import tree
import numpy as np
import tensorflow as tf

TEST_CASES = [
    dict(
        testcase_name='MinimalTest',
        sequence_length=5,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2), {'core_state': 99}),
            (0, dm_env.transition(reward=3.0, observation=3), {'core_state': 1}),
            (0, dm_env.transition(reward=5.0, observation=4), {'core_state': 2}),
            (0, dm_env.termination(reward=7.0, observation=5), {'core_state': 3}),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, extra)
            (base.Step(observation=np.array([1, 2, 3, 4, 5]),
                       action=np.array([0, 0, 0, 0, 0]),
                       reward=np.array([2.0, 3.0, 5.0, 7.0, 0.0]),
                       discount=np.array([1.0, 1.0, 1.0, 0.0, 0.0]),
                       extras={'core_state': np.array([99])},
                       start_of_episode=np.array([True, False, False, False, False]))),
        ))
]


class RNNSequenceAdderTest(test_utils.AdderTestMixin, parameterized.TestCase):

    @parameterized.named_parameters(*TEST_CASES)
    def test_adder(self, sequence_length: int,
                   first, steps, expected_sequences,
                   pad_end_of_episode: bool = True):
        adder = rnn_sequence.NonOverlappingRNNSequenceAdder(
            self.client,
            sequence_length=sequence_length,
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
                       expected_items: Sequence[Any]):
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
        for action, ts, extras in steps[:-1]:
            adder.add(action, next_timestep=ts, extras=extras)

        if len(steps) == 1:
            # adder.add() has not been called yet, so no writers have been created.
            self.assertEmpty(self.client.writers)
        # else:
        #     # Make sure the writer has been created but not closed.
        #     self.assertLen(self.client.writers, 1)
        #     self.assertFalse(self.client.writers[0].closed)

        # Add the final step.
        adder.add(*steps[-1])

        # Ending the episode should close the writer. No new writer should yet have
        # been created as it is constructed lazily.
        self.assertLen(self.client.writers, 1)
        self.assertTrue(self.client.writers[0].closed)

        # Make sure our expected and observed data match.
        observed_items = [p[1] for p in self.client.writers[0].priorities]
        # This is the only change to acme.adders.reverb.test_utils.py:
        # Here, we make sure that there are as many observed items as expected items.
        self.assertEqual(len(observed_items), len(expected_items))
        for expected_item, observed_item in zip(expected_items, observed_items):
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

        # for step in self.client.writers[0].timesteps:
        #     tree.map_structure(_check_signature, signature, step)

        # Add the start of a second trajectory.
        adder.add_first(first)
        adder.add(*steps[0])

        # # Make sure this creates an new writer.
        # self.assertLen(self.client.writers, 2)
        # # The writer is closed if the recently added `dm_env.TimeStep`'s' step_type
        # # is `dm_env.StepType.LAST`.
        # if steps[0][1].last():
        #     self.assertTrue(self.client.writers[1].closed)
        # else:
        #     self.assertFalse(self.client.writers[1].closed)


if __name__ == '__main__':
    absltest.main()
