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

"""Functions for making TensorFlow datasets for sampling from Reverb replay.

The functions implemented by this module closely resemble
acme.datasets.make_reverb_dataset(). Thus, most of the code and docstrings
was copied from this function.

However, make_reverb_fifo_sampler_dataset ensures that there is only one worker
per iterator for the ReplayDataset, since this might be of importance for agents
using a queue, where the order of elements drawn from the dataset is relevant.

Furthermore, make_reverb_rnn_sequence_fifo_sampler_dataset implements a dataset
for usage in recurrent agents that perform training on unrolled sequences but
only require the first recurrent state of the sequence. Consequently, this
function configures the dataset in such a way that sequences only include the
first recurrent state of a sequence, potentially saving a considerable amount of
RAM, especially with recurrent cores that have large states, such as a DNC
memory.

Edits made to the original script:
    -   Added the first paragraph of the docstring for
        make_reverb_fifo_sampler_dataset()
    -   Added the first paragraph of the docstring for
        make_reverb_rnn_sequence_fifo_sampler_dataset()
    -   Setting the default value of argument num_parallel_calls to 1,
        to save computational resources (threads).
    -   Passing num_workers_per_iterator=1 to the ReplayDataset constructor in
        both functions implemented by this module, to ensure compatability with
        agents using a queue.
    -   Exposing the deterministic argument of tf.dataset.interleave() as an
        argument to both functions implemented by this module, and setting its
        default value to True. Setting this to False may improve performance
        at the cost of determinism.
    -   Passing sequence_length=None and emit_timesteps=True to the construction
        of the ReplayDataset in make_reverb_rnn_sequence_fifo_sampler_dataset(),
        as well as manipulating the shape of core_state spec, so sequences
        contain only the first core_state of a sequence.
"""

from typing import Optional

from acme import specs
from acme import types
from acme.adders import reverb as adders
from acme.datasets import reverb as acme_datasets

import reverb
import tensorflow as tf
import tree


def make_reverb_fifo_sampler_dataset(
        client: reverb.TFClient,
        environment_spec: Optional[specs.EnvironmentSpec] = None,
        batch_size: Optional[int] = None,
        prefetch_size: Optional[int] = None,
        sequence_length: Optional[int] = None,
        extra_spec: Optional[types.NestedSpec] = None,
        transition_adder: bool = False,
        table: str = adders.DEFAULT_PRIORITY_TABLE,
        convert_zero_size_to_none: bool = False,
        num_parallel_calls: int = 1,  # 16
        using_deprecated_adder: bool = False,
        deterministic: bool = True
) -> tf.data.Dataset:
    """Makes a TensorFlow dataset.

    Ensures that there is only one worker per iterator for the ReplayDataset,
    since this might be of importance for agents using a queue,
    where the order of elements drawn from the dataset is relevant.

    We need to explicitly specify up-front the shapes and dtypes of all the
    Tensors that will be drawn from the dataset. We require that the action and
    observation specs are given. The reward and discount specs use reasonable
    defaults if not given. We can also specify a boolean `transition_adder` which
    if true will specify the spec as transitions rather than timesteps (i.e. they
    have a trailing state). Additionally an `extra_spec` parameter can be given
    which specifies "extra data".

    Args:
      client: A TFClient (or list of TFClients) for talking to a replay server.
      environment_spec: The environment's spec.
      batch_size: Optional. If specified the dataset returned will combine
        consecutive elements into batches. This argument is also used to determine
        the cycle_length for `tf.data.Dataset.interleave` -- if unspecified the
        cycle length is set to `tf.data.experimental.AUTOTUNE`.
      prefetch_size: How many batches to prefectch in the pipeline.
      sequence_length: Optional. If specified consecutive elements of each
        interleaved dataset will be combined into sequences.
      extra_spec: Optional. A possibly nested structure of specs for extras. Note
        that whether or not this is present changes the format of the data.
      transition_adder: Optional, defaults to False; whether the adder used with
        this dataset adds transitions.
      table: The name of the table to sample from replay (defaults to
        `adders.DEFAULT_PRIORITY_TABLE`).
      convert_zero_size_to_none: When True this will convert specs with shapes 0
        to None. This is useful for datasets that contain elements with different
        shapes for example `GraphsTuple` from the graph_net library. For example,
        `specs.Array((0, 5), tf.float32)` will correspond to a examples with shape
        `tf.TensorShape([None, 5])`.
      num_parallel_calls: Number of parallel threads creating ReplayDatasets to
        interleave.
      using_deprecated_adder: True if the adder used to generate the data is
        from acme/adders/reverb/deprecated.
      deterministic: Whether to use deterministic interleaving of dataset.

    Returns:
      A tf.data.Dataset that streams data from the replay server.
    """
    server_address: str = client._server_address  # pylint: disable=protected-access

    # This is the default that used to be set by reverb.TFClient.dataset().
    max_in_flight_samples_per_worker = 2 * batch_size if batch_size else 100

    def _make_dataset(unused_idx: tf.Tensor) -> tf.data.Dataset:
        if environment_spec is not None:
            shapes, dtypes = acme_datasets._spec_to_shapes_and_dtypes(
                # pylint: disable=protected-access
                transition_adder,
                environment_spec,
                extra_spec=extra_spec,
                sequence_length=sequence_length,
                convert_zero_size_to_none=convert_zero_size_to_none,
                using_deprecated_adder=using_deprecated_adder)
            dataset_ = reverb.ReplayDataset(
                server_address=server_address,
                table=table,
                dtypes=dtypes,
                shapes=shapes,
                max_in_flight_samples_per_worker=max_in_flight_samples_per_worker,
                num_workers_per_iterator=1,
                sequence_length=sequence_length,
                emit_timesteps=sequence_length is None)
        else:
            dataset_ = reverb.ReplayDataset.from_table_signature(
                server_address=server_address,
                table=table,
                max_in_flight_samples_per_worker=max_in_flight_samples_per_worker,
                num_workers_per_iterator=1,
                # ensures compatability with agents using a queue
                sequence_length=sequence_length,
                emit_timesteps=sequence_length is None)
        # Finish the pipeline: batch and prefetch.
        if batch_size:
            dataset_ = dataset_.batch(batch_size, drop_remainder=True)

        return dataset_

    # Create the dataset.
    dataset = tf.data.Dataset.range(1)
    dataset = dataset.interleave(
        map_func=_make_dataset,
        cycle_length=num_parallel_calls,
        num_parallel_calls=num_parallel_calls,
        deterministic=deterministic)

    if prefetch_size:
        dataset = dataset.prefetch(prefetch_size)

    return dataset


def make_reverb_rnn_sequence_fifo_sampler_dataset(
        client: reverb.TFClient,
        environment_spec: Optional[specs.EnvironmentSpec] = None,
        batch_size: Optional[int] = None,
        prefetch_size: Optional[int] = None,
        sequence_length: Optional[int] = None,
        extra_spec: Optional[types.NestedSpec] = None,
        transition_adder: bool = False,
        table: str = adders.DEFAULT_PRIORITY_TABLE,
        convert_zero_size_to_none: bool = False,
        num_parallel_calls: int = 1,  # 16
        using_deprecated_adder: bool = False,
        deterministic: bool = True
) -> tf.data.Dataset:
    """Makes a TensorFlow dataset.

    Implements a dataset for usage in recurrent agents that do training on
    unrolled sequences but only require the first recurrent state of the sequence.
    Consequently, this function configures the dataset in such a way that sequences
    only include the first recurrent state of the sequence, saving quite a lot of RAM,
    especially with recurrent cores that have large states, such as a DNC memory.

    We need to explicitly specify up-front the shapes and dtypes of all the
    Tensors that will be drawn from the dataset. We require that the action and
    observation specs are given. The reward and discount specs use reasonable
    defaults if not given. We can also specify a boolean `transition_adder` which
    if true will specify the spec as transitions rather than timesteps (i.e. they
    have a trailing state). Additionally an `extra_spec` parameter can be given
    which specifies "extra data".

    Args:
      client: A TFClient (or list of TFClients) for talking to a replay server.
      environment_spec: The environment's spec.
      batch_size: Optional. If specified the dataset returned will combine
        consecutive elements into batches. This argument is also used to determine
        the cycle_length for `tf.data.Dataset.interleave` -- if unspecified the
        cycle length is set to `tf.data.experimental.AUTOTUNE`.
      prefetch_size: How many batches to prefectch in the pipeline.
      sequence_length: Optional. If specified consecutive elements of each
        interleaved dataset will be combined into sequences.
      extra_spec: Optional. A possibly nested structure of specs for extras. Note
        that whether or not this is present changes the format of the data.
      transition_adder: Optional, defaults to False; whether the adder used with
        this dataset adds transitions.
      table: The name of the table to sample from replay (defaults to
        `adders.DEFAULT_PRIORITY_TABLE`).
      convert_zero_size_to_none: When True this will convert specs with shapes 0
        to None. This is useful for datasets that contain elements with different
        shapes for example `GraphsTuple` from the graph_net library. For example,
        `specs.Array((0, 5), tf.float32)` will correspond to a examples with shape
        `tf.TensorShape([None, 5])`.
      num_parallel_calls: Number of parallel threads creating ReplayDatasets to
        interleave.
      using_deprecated_adder: True if the adder used to generate the data is
        from acme/adders/reverb/deprecated.
      deterministic: Whether to use deterministic interleaving of dataset.

    Returns:
      A tf.data.Dataset that streams data from the replay server.
    """
    server_address: str = client._server_address  # pylint: disable=protected-access

    # This is the default that used to be set by reverb.TFClient.dataset().
    max_in_flight_samples_per_worker = 2 * batch_size if batch_size else 100

    def _make_dataset(unused_idx: tf.Tensor) -> tf.data.Dataset:
        if environment_spec is not None:
            shapes, dtypes = acme_datasets._spec_to_shapes_and_dtypes(
                # pylint: disable=protected-access
                transition_adder,
                environment_spec,
                extra_spec=extra_spec,
                sequence_length=sequence_length,
                convert_zero_size_to_none=convert_zero_size_to_none,
                using_deprecated_adder=using_deprecated_adder)
            if sequence_length and 'core_state' in shapes[-1]:
                # Manipulate shape of core_state spec,
                # so sequences contain only the first core_state of a sequence.
                shapes[-1].update(core_state=tree.map_structure(
                    lambda x: tf.TensorShape([1, *x.shape]),
                    extra_spec['core_state']))
            dataset_ = reverb.ReplayDataset(
                server_address=server_address,
                table=table,
                dtypes=dtypes,
                shapes=shapes,
                max_in_flight_samples_per_worker=max_in_flight_samples_per_worker,
                num_workers_per_iterator=1,
                sequence_length=None,
                emit_timesteps=True)
        else:
            dataset_ = reverb.ReplayDataset.from_table_signature(
                server_address=server_address,
                table=table,
                max_in_flight_samples_per_worker=max_in_flight_samples_per_worker,
                num_workers_per_iterator=1,
                # ensures compatability with agents using a queue
                sequence_length=None,
                emit_timesteps=True)
        # Finish the pipeline: batch and prefetch.
        if batch_size:
            dataset_ = dataset_.batch(batch_size, drop_remainder=True)

        return dataset_

    # Create the dataset.
    dataset = tf.data.Dataset.range(1)
    dataset = dataset.interleave(
        map_func=_make_dataset,
        cycle_length=num_parallel_calls,
        num_parallel_calls=num_parallel_calls,
        deterministic=deterministic)

    if prefetch_size:
        dataset = dataset.prefetch(prefetch_size)

    return dataset
