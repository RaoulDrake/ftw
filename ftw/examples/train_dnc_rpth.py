# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example script to train the DNC or RPTH on a repeated copy task.

Taken from the original DNC repository at https://github.com/deepmind/dnc and modified
to be compatible to TensorFlow version 2.x. Furthermore, the Recurrent processing with
temporal hierarchy (RPTH) module, as featured in the FTW paper by Jaderberg et al. (2019),
can also be trained on the same task. By default, the RPTH module will be selected.
Behaviour and several parameters, such as memory size can be modified via Flags.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow as tf
import sonnet as snt
import tree

from ftw.tf.networks.dnc import dnc, access
from ftw.tf.networks.dnc import repeat_copy
from ftw.tf.networks import recurrence

FLAGS = flags.FLAGS

# Model parameters
flags.DEFINE_string("dnc_or_rpth", "rpth", "Whether to use DNC or RPTH.")
flags.DEFINE_integer("hidden_size", 64, "Size of LSTM hidden layer.")
flags.DEFINE_integer("memory_size", 16, "The number of memory slots.")
flags.DEFINE_integer("word_size", 16, "The width of each memory slot.")
flags.DEFINE_integer("num_write_heads", 1, "Number of memory write heads.")
flags.DEFINE_integer("num_read_heads", 4, "Number of memory read heads.")
flags.DEFINE_integer("clip_value", 20, "Maximum absolute value of controller and dnc outputs.")

# Optimizer parameters.
flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
flags.DEFINE_float("learning_rate", 1e-4, "Optimizer learning rate.")
flags.DEFINE_float("optimizer_epsilon", 1e-10, "Epsilon used for RMSProp optimizer.")

# Task parameters
flags.DEFINE_integer("batch_size", 16, "Batch size for training.")
flags.DEFINE_integer("num_bits", 4, "Dimensionality of each vector to copy")
flags.DEFINE_integer("min_length", 1,
                     "Lower limit on number of vectors in the observation pattern to copy")
flags.DEFINE_integer("max_length", 2,
                     "Upper limit on number of vectors in the observation pattern to copy")
flags.DEFINE_integer("min_repeats", 1, "Lower limit on number of copy repeats.")
flags.DEFINE_integer("max_repeats", 2, "Upper limit on number of copy repeats.")

# Training options.
flags.DEFINE_integer("num_training_iterations", 100000, "Number of iterations to train for.")
flags.DEFINE_integer("report_interval", 100, "Iterations between reports (samples, valid loss).")
flags.DEFINE_string("checkpoint_dir", "/tmp/tf/dnc", "Checkpointing directory.")
flags.DEFINE_integer("checkpoint_interval", -1, "Checkpointing step interval.")


class DummyRPTH(snt.RNNCore):

    def __init__(self, period, hidden_size, output_size, clip_value,
                 access_config, name='dummy_rpth'):
        super().__init__(name=name)
        self._rpth = recurrence.RPTH(
            period=period, hidden_size=hidden_size,
            num_dimensions=output_size,
            dnc_clip_value=clip_value,
            shared_memory=access.MemoryAccess(**access_config),
            init_scale=0.1
        )
        self._linear = snt.Linear(output_size)
        self._clip_value = clip_value

    def _clip_if_enabled(self, x):
        if self._clip_value > 0:
            return tf.clip_by_value(x, clip_value_min=-self._clip_value,
                                    clip_value_max=self._clip_value)
        else:
            return x

    def __call__(self, inputs, state):
        core_out, core_state = self._rpth(inputs, state)
        out = self._linear(core_out.z)
        out = self._clip_if_enabled(out)
        return out, core_state

    def initial_state(self, batch_size: int, **unused_kwargs):
        return self._rpth.initial_state(batch_size)


def train(num_training_iterations, report_interval):
    """Trains the DNC and periodically reports the loss."""
    dataset = repeat_copy.RepeatCopy(FLAGS.num_bits, FLAGS.batch_size,
                                     FLAGS.min_length, FLAGS.max_length,
                                     FLAGS.min_repeats, FLAGS.max_repeats)

    # Set up DNC.
    access_config = {
        "memory_size": FLAGS.memory_size,
        "word_size": FLAGS.word_size,
        "num_reads": FLAGS.num_read_heads,
        "num_writes": FLAGS.num_write_heads,
    }
    controller_config = {
        "hidden_size": FLAGS.hidden_size,
    }
    clip_value = FLAGS.clip_value

    optimizer = snt.optimizers.RMSProp(
        FLAGS.learning_rate, epsilon=FLAGS.optimizer_epsilon)

    if FLAGS.dnc_or_rpth == "dnc":
        core = dnc.DNC(access_config, controller_config, dataset.target_size, clip_value)
    else:
        core = DummyRPTH(
            period=2, hidden_size=controller_config['hidden_size'],
            output_size=dataset.target_size,
            clip_value=clip_value,
            access_config=access_config)

    @tf.function
    def train_step():
        dataset_tensors = dataset()
        with tf.GradientTape() as tape:
            initial_state = core.initial_state(FLAGS.batch_size)
            output_logits, _ = snt.dynamic_unroll(
                core=core,
                input_sequence=dataset_tensors.observations,
                initial_state=initial_state)
            # Used for visualization.
            output = tf.round(
                tf.expand_dims(dataset_tensors.mask, -1) * tf.sigmoid(output_logits))

            train_loss = dataset.cost(output_logits, dataset_tensors.target,
                                      dataset_tensors.mask)
        grads = tape.gradient(train_loss, core.trainable_variables)
        grads, _ = tf.clip_by_global_norm(
            grads, FLAGS.max_grad_norm)
        optimizer.apply(grads, core.trainable_variables)
        return output, train_loss, dataset_tensors

    # Train.
    start_iteration = 0
    total_loss = 0

    for train_iteration in range(start_iteration, num_training_iterations):
        output, loss, ds_tensors = train_step()
        total_loss += loss

        if (train_iteration + 1) % report_interval == 0:
            dataset_tensors_np = tree.map_structure(lambda t: t.numpy(), ds_tensors)
            output_np = output.numpy()
            dataset_string = dataset.to_human_readable(dataset_tensors_np,
                                                       output_np)
            tf.compat.v1.logging.info("%d: Avg training loss %f.\n%s",
                                      train_iteration, total_loss / report_interval,
                                      dataset_string)
            total_loss = 0


def main(_):
    tf.compat.v1.logging.set_verbosity(3)  # Print INFO log messages.
    train(FLAGS.num_training_iterations, FLAGS.report_interval)


if __name__ == "__main__":
    app.run(main)
