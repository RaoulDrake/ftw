from typing import Callable

import tensorflow as tf
import sonnet as snt


class PixelControl(snt.Module):
    """Module that produces a pixel control output (i.e. Q-values) from a hidden state input.

    This module implements the Pixel Control module from the FTW paper.

    Thus, it produces an output of shape [batch_size, 20, 20, num_actions], representing a grid of 20 x 20 cells,
    each representing a 5 x 5 pixel area, covering a pixel area of altogether 80 x 80 pixels
    (= (20 cells x 5 pixels) x (20 cells x 5 pixels)).

    Consequently, the output produced by this module can only be used for pixel control loss calculation if the
    observations input to the pixel control loss function is of shape [sequence_length, batch_size, 80, 80, 3]
    (Pixel control only supports RGB Pixel observations).

    Recommended usage is to have 84 x 84 x 3 RGB pixel observations as input to this module and
    to crop these observations to the central 80 x 80 pixel area for loss calculation,
    as done by the FTW and UNREAL agents.
    """

    def __init__(self, num_actions: int,
                 activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
                 name: str = 'pixel_control'):
        """Initializes the PixelControl module.

        Args:
            num_actions: number of actions in discrete action space
            activation: activation function to be used (after linear and deconvolutional layer)
            name: name for the module
        """
        super(PixelControl, self).__init__(name=name)
        self._num_actions = num_actions
        self._activation = activation

        self._linear = snt.Linear(32 * 7 * 7, name='linear')
        self._deconv = snt.Conv2DTranspose(
            output_channels=32, kernel_shape=9, padding='SAME', name='deconv',
            stride=3, output_shape=(20, 20))
        self._value = snt.Conv2DTranspose(
            output_channels=1, kernel_shape=4, padding='SAME', stride=1,
            name='value')
        self._advantage = snt.Conv2DTranspose(
            output_channels=self._num_actions, kernel_shape=4, padding='SAME', stride=1,
            name='advantage')

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """Produces Pixel Control Q-values from a (batched) hidden state input.

        Args:
            inputs: (Batched) pixel observations, in the form of a tf.Tensor or an observation_action_reward.OAR
                namedtuple.

        Returns:
            pixel_control_q_vals: (Batched) Pixel control Q-values of shape [B, 20, 20, num_actions].
        """
        linear = self._linear(inputs)
        linear = self._activation(linear)
        linear = tf.reshape(linear, [-1, 7, 7, 32])  # reshape to [B, 7, 7, 32]

        deconv = self._deconv(linear)  # output shape [B, 20, 20, 32]
        deconv = self._activation(deconv)

        value = self._value(deconv)  # output shape [B, 20, 20, 1]
        advantage = self._advantage(deconv)  # output shape [B, 20, 20, num_actions]

        # decoding into Q-Values using the dueling parametrization (Wang et al., 2016)
        advantage_mean = tf.reduce_mean(advantage, axis=-1, keepdims=True)
        q = value + (advantage - advantage_mean)

        return q


class RNNPixelControlNetwork(snt.RNNCore):
    """Module that produces a Pixel control output (i.e. Q-values) from a pixel observations input.

    This module implements the Pixel control module from the FTW paper and wraps it together with a (possibly shared)
    visual embedding module (= embed) and a (possibly shared) recurrent core (= core).
    Thus, it produces an output of shape [batch_size, 20, 20, num_actions], representing a grid of 20 x 20 cells,
    each representing a 5 x 5 pixel area, covering a pixel area of altogether 80 x 80 pixels
    (= (20 cells x 5 pixels) x (20 cells x 5 pixels)).
    Consequently, the output produced by this module can only be used for Pixel control loss calculation if the
    observations input to the Pixel control loss function is of shape [sequence_length, batch_size, 80, 80, 3]
    (Pixel control only supports RGB Pixel observations).
    """

    def __init__(self,
                 embed: snt.Module,
                 core: snt.RNNCore,
                 num_actions: int,
                 activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
                 name: str = 'rnn_pixel_control_network'):
        """Initializes the RNNPixelControlNetwork module.

        Args:
            embed: Visual embedding module (of type sonnet.Module) to transform observations into an embedding,
                e.g. FtwTorso.
            core: Recurrent core (of type sonnet.RNNCore) to transform embedding into input for PixelControl module,
                e.g. RPTH.
            num_actions: number of actions in discrete action space.
            activation: activation function to be used in PixelControl module (after linear and deconvolutional layer)
            name: name for the module.
        """
        super().__init__(name=name)
        self._embed = embed
        self._core = core
        self._pixel_control = PixelControl(num_actions=num_actions, activation=activation)

    def __call__(self, inputs, state):
        """Produces Pixel control Q-values from a (batched) pixel observations input.

        Args:
            inputs: (Batched) pixel observations, in the form of a tf.Tensor or an observation_action_reward.OAR
                namedtuple.

        Returns:
            pixel_control_q_vals: (Batched) Pixel control Q-values of shape [B, 20, 20, num_actions].
        """
        embeddings = self._embed(inputs)
        embeddings, new_states = self._core(embeddings, state)
        pixel_control_q_vals = self._pixel_control(embeddings)

        return pixel_control_q_vals

    def unroll(self, inputs, state):
        """Unrolls the module over a sequence of pixel observation inputs and produces Pixel Control Q-values.

        Args:
            inputs: Sequence of (batched) Pixel observations, in the form of a tf.Tensor or an
                observation_action_reward.OAR namedtuple.

        Returns:
            pixel_control_q_vals: Sequence of (batched) Pixel control Q-values with shape [T, B, 20, 20, num_actions].
        """
        embeddings = snt.BatchApply(self._embed)(inputs)
        embeddings, new_states = snt.static_unroll(
            core=self._core, input_sequence=embeddings, initial_state=state)
        pixel_control_q_vals = snt.BatchApply(self._pixel_control)(embeddings)

        return pixel_control_q_vals

    def initial_state(self, batch_size: int, **kwargs):
        """Returns the initial state of the recurrent core."""
        return self._core.initial_state(batch_size, **kwargs)


# TODO: implement decomposed Pixel Control class


class RewardPrediction(snt.Module):
    """Module that produces a reward prediction output from a hidden state tensor.

    This module implements the Reward prediction module from the FTW paper and wraps it together with a
    (possibly shared) embedding module (= embed). Thus, its output is a logits tensor, representing the
    log-probabilities for the 3 categories to predict (zero reward, negative reward, positive reward).
    """

    def __init__(self,
                 hidden_size: int = 128,
                 activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
                 name='reward_prediction'):
        """Initializes the RewardPrediction module.

        Args:
            hidden_size: size of hidden linear layer
            activation: activation function to be used (between linear and logits layer)
            name: name for the module.
        """
        super(RewardPrediction, self).__init__(name=name)

        self._linear = snt.Linear(hidden_size, name='linear')
        self._logits = snt.Linear(3, name='logits')
        layers = [self._linear, activation, self._logits]
        self._sequential = snt.Sequential(layers)

    def __call__(self, inputs):
        """Produces reward prediction output (in the form of logits) from a hidden state input.

        Args:
            inputs: (Batched) hidden state input, in the form of a tf.Tensor.

        Returns:
            logits: (Batched) logits of shape [B, 3], representing log-probabilities for the 3 categories to predict
                (zero, negative and positive reward).
        """
        logits = self._sequential(inputs)

        return logits


class RewardPredictionNetwork(snt.Module):
    """Module that produces a reward prediction output from an observations input.

    This module implements the Reward prediction module from the FTW paper and wraps it together with a
    (possibly shared) embedding module (= embed). Thus, its output is a logits tensor, representing the
    log-probabilities for the 3 categories to predict (zero reward, negative reward, positive reward).
    """

    def __init__(self,
                 embed: snt.Module,
                 hidden_size: int = 128,
                 activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
                 name='reward_prediction_network'):
        """Initializes the RewardPredictionNetwork module.

        Args:
            embed: Embedding module (of type sonnet.Module) to transform observations into an embedding, e.g. FtwTorso.
            hidden_size: size of hidden linear layer
            activation: activation function to be used in RewardPrediction module (between linear and logits layer)
            name: name for the module.
        """
        super().__init__(name=name)
        self._embed = embed
        self._reward_prediction = RewardPrediction(hidden_size=hidden_size, activation=activation)

    def __call__(self, inputs) -> tf.Tensor:
        """Produces reward prediction output (in the form of logits) from an observations input.

        Logits represent the log-probabilities for the 3 categories to predict (zero, negative and positive reward).

        Contrary to usual sonnet modules, this module expects inputs of shape [B, T, D], where B is the batch size,
        T is the sequence dimension over which to concatenate before predicting, and D is the input feature dimension.

        Since Reward Prediction in the FTW and UNREAL papers receives as input a batch of sequences of embeddings,
        we concatenate the embedding sequences per batch (by using snt.flatten(), which flattens the input while
        preserving the first (i.e. batch) dimension), as in the UNREAL paper.

        Args:
            inputs: Batch of sequences of observations, in the form of a tf.Tensor.

        Returns:
            logits: (Batched) logits, representing log-probabilities for the 3 categories to predict
                (zero, negative and positive reward).
        """
        embeddings = snt.BatchApply(self._embed)(inputs)
        embeddings = snt.flatten(embeddings)
        logits = self._reward_prediction(embeddings)

        return logits
