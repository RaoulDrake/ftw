from typing import Callable, Sequence, Tuple
import tensorflow as tf
import sonnet as snt


class FtwTorso(snt.Module):
    """Visual embedding module as used in the FTW paper.

    See also the FTW paper for more information, especially Figure S11 of the supplementary material.
    """

    def __init__(self,
                 conv_filters: Sequence[Tuple[int, int, int]] = ((32, 8, 4), (64, 4, 2)),
                 residual_filters: Sequence[Tuple[int, int, int]] = ((64, 3, 1), (64, 3, 1)),
                 hidden_size: int = 256,
                 activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
                 activate_last: bool = False,
                 name: str = 'ftw_torso'):
        """Initializes the FtwTorso module.

        Args:
              conv_filters: Sequence of int triples (num_channels, kernel_size, stride) indicating the number of
                channels, the kernel size (also called filter size) and stride for each (non-residual) convolutional
                layer in the sequence.
              residual_filters: Sequence of int triples (num_channels, kernel_size, stride) indicating the number of
                channels, the kernel size (also called filter size) and stride for each residual convolutional
                layer in the sequence.
              hidden_size: Size of the final output layer.
              activation: Activation function to be used between layers.
              activate_last: Whether or not to pass the output of the final layer through the activation function given
                by activation.
              name: Name for the module.

        Raises:
            ValueError: If shapes and/or types of constructor arguments do not match expected shapes and types.
        """

        super(FtwTorso, self).__init__(name=name)
        for arg_name, filters_description in [('conv_filters', conv_filters), ('residual_filters', residual_filters)]:
            if not(isinstance(filters_description, Sequence) and (all([
                len(layer_triple) == 3 and all([isinstance(arg, int) for arg in layer_triple])
                for layer_triple in filters_description]))
            ):
                raise ValueError(f"{arg_name} must be a sequence of int triples (num_channels, kernel_size, stride) "
                                 f"indicating the number of channels, the kernel size (also called filter size) and "
                                 f"stride for each layer in the sequence, but {arg_name} given ({filters_description}) "
                                 f"was of type {type(filters_description)}.")
        if not isinstance(hidden_size, int):
            raise ValueError(f"hidden_size must be of type int but was of type {type(hidden_size)}.")
        if not isinstance(activate_last, bool):
            raise ValueError(f"activate_last must be of type bool but was of type {type(activate_last)}.")

        # Internalize activation-specific args.
        self._activation = activation
        self._activate_last = activate_last

        # Convolution Layers.
        conv_layers = []
        for i, (num_ch, kernel_size, stride) in enumerate(conv_filters):  # [:-1]):
            conv_layers.append(
                snt.Conv2D(num_ch, kernel_size, stride=stride, padding='SAME', name='conv2d_%d' % i))
            conv_layers.append(self._activation)
        self._conv_layers = snt.Sequential(conv_layers, name='conv_tower')

        # Residual layers.
        self._residual_layers = []
        for i, (num_ch, kernel_size, stride) in enumerate(residual_filters):
            self._residual_layers.append(
                snt.Conv2D(num_ch, kernel_size, stride=stride, padding='SAME', name='residual_conv2d_%d' % i))

        # Last layer:
        self._linear = snt.Linear(hidden_size, name='linear')

    def __call__(self, inputs) -> tf.Tensor:
        """ConvNet with residual connections and a linear output.

        Implementation specific to Deepmind's FTW paper (Jaderberg et al.).
        See also the FTW paper for more information, especially Figure S11 of the supplementary material.

        Args:
            inputs: Pixel observations.

        Returns:
            conv_out: Embedding of pixel observations.

        """

        # Forward pass through convolution layer(s).
        conv_out = self._conv_layers(inputs)

        # Forward pass through residual layer(s).
        for j, layer in enumerate(self._residual_layers):
            block_input = conv_out
            if j > 0:
                conv_out = self._activation(conv_out)
            conv_out = layer(conv_out)
            conv_out += block_input

        conv_out = self._activation(conv_out)
        # Flatten the output to [B, D].
        conv_out = snt.Flatten()(conv_out)

        # Forward pass through last (linear) layer
        conv_out = self._linear(conv_out)
        if self._activate_last:
            conv_out = self._activation(conv_out)

        return conv_out
