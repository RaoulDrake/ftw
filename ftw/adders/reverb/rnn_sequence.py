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

This implements adders which add sequences or partial trajectories.
"""

from typing import Optional

from acme.adders.reverb import base
from acme.adders.reverb import utils
from acme.tf import utils as tf2_utils

import reverb
import tree

CORE_STATE = 'core_state'


class NonOverlappingRNNSequenceAdder(base.ReverbAdder):
    """An adder which adds non-overlapping sequences of fixed length.

    This adder is based on acme.adders.reverb.SequenceAdder, with the following modification:
    Enables adding sequences, where only the first recurrent core_state of the sequence is included.
    This saves RAM and can be used for any recurrent agent that is trained on unrolled sequences,
    if no overlapping of sequences is required.

    Must be used in combination with ftw.datasets.make_reverb_rnn_sequence_fifo_sampler_dataset().
    """

    def __init__(
            self,
            client: reverb.Client,
            sequence_length: int,
            delta_encoded: bool = False,
            chunk_length: Optional[int] = None,
            priority_fns: Optional[base.PriorityFnMapping] = None,
            pad_end_of_episode: bool = True,
    ):
        """Makes a SequenceAdder instance.

        Args:
          client: See docstring for BaseAdder.
          sequence_length: The fixed length of sequences we wish to add.
          delta_encoded: If `True` (False by default) enables delta encoding, see
            `Client` for more information.
          chunk_length: Number of timesteps grouped together before delta encoding
            and compression. See `Client` for more information.
          priority_fns: See docstring for BaseAdder.
          pad_end_of_episode: If True (default) then upon end of episode the current
            sequence will be padded (with observations, actions, etc... whose values
            are 0) until its length is `sequence_length`. If False then the last
            sequence in the episode may have length less than `sequence_length`.
        """
        super().__init__(
            client=client,
            buffer_size=sequence_length,
            max_sequence_length=sequence_length,
            delta_encoded=delta_encoded,
            chunk_length=chunk_length,
            priority_fns=priority_fns)

        self._step = 0
        self._pad_end_of_episode = pad_end_of_episode

    def reset(self):
        self._step = 0
        super().reset()

    def _write(self):
        # Append the previous step and increment number of steps written.
        # self._writer.append(self._buffer[-1])
        self._step += 1
        self._maybe_add_priorities()

    def _write_last(self):
        # Create a final step and a step full of zeros.
        final_step = utils.final_step_like(self._buffer[0], self._next_observation)
        zero_step = tree.map_structure(utils.zeros_like, final_step)

        # Append the final step.
        self._buffer.append(final_step)
        # self._writer.append(final_step)
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
                if self._step % self._max_sequence_length != 0:
                    padding = self._max_sequence_length - (
                            (self._step - self._max_sequence_length) % self._max_sequence_length)

            # Pad with zeros to get a full sequence.
            for _ in range(padding):
                self._buffer.append(zero_step)
                # self._writer.append(zero_step)
                self._step += 1

        # Write priorities for the sequence.
        self._maybe_add_priorities()

    def _maybe_add_priorities(self):
        if not (
                # Write the first time we hit the max sequence length...
                self._step == self._max_sequence_length or
                # ... or every `period`th time after hitting max length.
                (self._step > self._max_sequence_length and
                 (self._step - self._max_sequence_length) % self._max_sequence_length == 0)):
            return

        # Compute priorities for the buffer.
        steps = list(self._buffer)
        table_priorities = utils.calculate_priorities(self._priority_fns, steps)

        data = base.Step(*tf2_utils.stack_sequence_fields(steps))
        if data.extras:
            new_extras = {}
            for extra_key, extra_value in data.extras.items():
                if extra_key != CORE_STATE:
                    new_extras[extra_key] = extra_value
                else:
                    # Extract first core_state of sequence and
                    # add a batch dimension to this extracted first core_state.
                    new_extras[extra_key] = tree.map_structure(
                        lambda t: tf2_utils.add_batch_dim(t[0]), extra_value)
            data = data._replace(extras=new_extras)
        self._writer.append(data)

        # Create a prioritized item for each table.
        for table_name, priority in table_priorities.items():
            self._writer.create_item(table_name, 1, priority)
