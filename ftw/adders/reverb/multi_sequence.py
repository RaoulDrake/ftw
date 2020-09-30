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

"""Sequence adders.

This implements adders which add multiple sequences.
"""

from typing import Optional, Mapping, Callable

from acme.adders.reverb import base
from acme.adders.reverb import utils

import reverb
import tree

ShouldInsertFn = Callable[[base.PriorityFnInput], bool]
ShouldInsertFnMapping = Mapping[str, ShouldInsertFn]


class MultiSequenceAdder(base.ReverbAdder):
    """An adder which adds multiple sequences of fixed lengths.

    This class is based on acme.adders.reverb.SequenceAdder, with the following enhancements:
        -   supports multiple sequences, for which the
            sequence lengths and periods may differ for each sequence.
        -   supports conditional inserting of sequences,
            via the should_insert_fns argument of the constructor.
    Furthermore, this class implements a similar bugfix (related to padding) as ftw.adders.reverb.SequenceAdder.
    """

    def __init__(
            self,
            client: reverb.Client,
            sequence_lengths: Mapping[str, int],
            periods: Mapping[str, int],
            delta_encoded: bool = False,
            chunk_length: Optional[int] = None,
            priority_fns: Optional[base.PriorityFnMapping] = None,
            should_insert_fns: Optional[ShouldInsertFnMapping] = None,
            pad_end_of_episode: bool = True,
    ):
        """Makes a SequenceAdder instance.

        Args:
          client: See docstring for BaseAdder.
          sequence_lengths: Dictionary mapping table names
            to the fixed length of sequences we wish to add for that table.
          periods: Dictionary mapping table names
            to the period with which we add sequences to that table. If less than
            sequence_length, overlapping sequences are added. If equal to
            sequence_length, sequences are exactly non-overlapping.
          delta_encoded: If `True` (False by default) enables delta encoding, see
            `Client` for more information.
          chunk_length: Number of timesteps grouped together before delta encoding
            and compression. See `Client` for more information.
          priority_fns: See docstring for BaseAdder.
          should_insert_fns: Dictionary of functions for each sequence (table),
            taking a PriorityFnInput NamedTuple as input and returning a bool indicating
            whether the sequence should be added to the corresponding table.
          pad_end_of_episode: If True (default) then upon end of episode the current
            sequence will be padded (with observations, actions, etc... whose values
            are 0) until its length is `sequence_length`. If False then the last
            sequence in the episode may have length less than `sequence_length`.

        Raises:
          ValueError: If invalid or conflicting arguments are passed to the constructor.
        """
        if len(sequence_lengths.items()) != len(periods.items()):
            raise ValueError('MultiSequenceAdder: '
                             'sequence_lengths and periods must have matching length.')
        if sorted(sequence_lengths.keys()) != sorted(periods.keys()):
            raise ValueError('MultiSequenceAdder: '
                             'sequence_lengths and periods must have matching keys.')

        if should_insert_fns:
            if len(sequence_lengths.items()) != len(should_insert_fns.items()):
                raise ValueError('MultiSequenceAdder: should_insert_fns and sequence_lengths/periods '
                                 'must have matching length.')
            if sorted(sequence_lengths.keys()) != sorted(should_insert_fns.keys()):
                raise ValueError('MultiSequenceAdder: should_insert_fns and sequence_lengths/periods '
                                 'must have matching keys.')
            should_insert_fns = dict(should_insert_fns)
        else:
            should_insert_fns = {}
            for key in sequence_lengths.keys():
                should_insert_fns[key] = lambda x: True

        if priority_fns:
            if len(sequence_lengths.items()) != len(priority_fns.items()):
                raise ValueError('MultiSequenceAdder: priority_fns and sequence_lengths/periods '
                                 'must have matching length.')
            if sorted(sequence_lengths.keys()) != sorted(priority_fns.keys()):
                raise ValueError('MultiSequenceAdder: priority_fns and sequence_lengths/periods '
                                 'must have matching keys.')
            priority_fns = dict(priority_fns)
        else:
            priority_fns = {}
            for key in sequence_lengths.keys():
                priority_fns[key] = lambda x: 1.

        self._should_insert_fns = should_insert_fns

        self._max_sequence_length = max(sequence_lengths.values())
        self._max_period = max(periods.values())

        super().__init__(
            client=client,
            buffer_size=self._max_sequence_length,
            max_sequence_length=self._max_sequence_length,
            delta_encoded=delta_encoded,
            chunk_length=chunk_length,
            priority_fns=priority_fns)

        self._sequence_lengths = sequence_lengths
        self._periods = periods
        self._sequence_lengths_periods_and_insert_fns = {}
        for key, item in sequence_lengths.items():
            self._sequence_lengths_periods_and_insert_fns[key] = (
                item, periods[key], self._should_insert_fns[key])
        self._step = 0
        self._pad_end_of_episode = pad_end_of_episode

    def reset(self):
        self._step = 0
        super().reset()

    def _write(self):
        # Append the previous step and increment number of steps written.
        self._writer.append(self._buffer[-1])
        self._step += 1
        self._maybe_add_priorities()

    def _write_last(self):
        # Create a final step and a step full of zeros.
        final_step = utils.final_step_like(self._buffer[0], self._next_observation)
        zero_step = tree.map_structure(utils.zeros_like, final_step)

        # Append the final step.
        self._buffer.append(final_step)
        self._writer.append(final_step)
        self._step += 1

        # NOTE: this always pads to the fixed length. but this is not equivalent to
        # the old Padded sequence adder.

        if self._pad_end_of_episode:
            # Determine how much padding to add. This makes sure that we add (zero)
            # data until the next time we would write a sequence.
            if self._step <= self._max_sequence_length:
                padding = self._max_sequence_length - self._step
            else:
                padding = 0
                if (self._step - self._max_sequence_length) % self._max_period != 0:
                    padding = self._max_period - ((self._step - self._max_sequence_length) % self._max_period)

            # Pad with zeros to get a full sequence.
            for _ in range(padding):
                # Write priorities for the sequence.
                self._maybe_add_priorities()
                self._buffer.append(zero_step)
                self._writer.append(zero_step)
                self._step += 1

        # Write priorities for the sequence.
        self._maybe_add_priorities()

    def _maybe_add_priorities(self):
        for (table_name,
             (max_sequence_length, period, should_insert_fn)
             ) in self._sequence_lengths_periods_and_insert_fns.items():
            if not (
                    # Write the first time we hit the max sequence length...
                    self._step == max_sequence_length or
                    # ... or every `period`th time after hitting max length.
                    (self._step > max_sequence_length and
                     (self._step - max_sequence_length) % period == 0)):
                pass
            else:
                # Compute priorities for the buffer.
                # We only create an item if:
                #   - should_insert_fn(steps) evaluates to True,
                #   - the steps to add are not made up entirely of zero steps (--> is_all_zero_steps),
                #   - the sequence is not one that was padded solely as a side effect of another
                #     (required) padding for a different table_name/max_sequence_length/period combination
                #     (--> is_already_done).
                steps = list(self._buffer)[len(self._buffer) - max_sequence_length:]
                padding_checks_ok = True
                if self._pad_end_of_episode:
                    final_step = utils.final_step_like(self._buffer[0], self._next_observation)
                    zero_step = tree.map_structure(utils.zeros_like, final_step)
                    is_all_zero_steps = all([
                        all(tree.map_structure(lambda x, y: x == y, step, zero_step)) for step in steps])
                    is_already_done = (self._step > max_sequence_length and
                                       all([all(tree.map_structure(lambda x, y: x == y, step, zero_step))
                                            for step in steps[-period:]]))
                    padding_checks_ok = not is_all_zero_steps and not is_already_done
                if should_insert_fn(steps) and padding_checks_ok:
                    num_steps = len(steps)
                    table_priorities = utils.calculate_priorities(
                        {table_name: self._priority_fns[table_name]}, steps)

                    # Create a prioritized item for the table.
                    self._writer.create_item(table_name, num_steps, table_priorities[table_name])
