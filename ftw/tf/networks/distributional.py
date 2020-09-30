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

"""This module refactors the original MultivariateNormalDiagHead class.

Specifically, the class is split into two classes, so that one of the resulting classes
(MultivariateNormalDiagLocScaleHead) outputs mean and scale of the distribution
(for use in the Recurrent processing with temporal hierarchy module), while the other
(MultivariateNormalDiagHead) returns a tensorflow_probability distribution object, as
in the original acme module.
"""

from typing import Tuple, Union, Optional

import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
snt_init = snt.initializers
Initializer = Optional[Union[snt_init.Initializer, tf.initializers.Initializer]]


class MultivariateNormalDiagHead(snt.Module):
    """Module that produces a multivariate normal distribution using tfd.Independent or tfd.MultivariateNormalDiag."""

    def __init__(
            self,
            num_dimensions: int,
            init_scale: float = 0.3,
            min_scale: float = 1e-6,
            tanh_mean: bool = False,
            fixed_scale: bool = False,
            use_tfd_independent: bool = False,
            w_init: Initializer = tf.initializers.VarianceScaling(1e-4),
            b_init: Initializer = tf.initializers.Zeros()):
        """Initialization.

        Args:
          num_dimensions: Number of dimensions of MVN distribution.
          init_scale: Initial standard deviation.
          min_scale: Minimum standard deviation.
          tanh_mean: Whether to transform the mean (via tanh) before passing it to
            the distribution.
          fixed_scale: Whether to use a fixed variance.
          use_tfd_independent: Whether to use tfd.Independent or
            tfd.MultivariateNormalDiag class
          w_init: Initialization for linear layer weights.
          b_init: Initialization for linear layer biases.
        """
        super().__init__(name='MultivariateNormalDiagHead')
        self._loc_scale = MultivariateNormalDiagLocScaleHead(
            num_dimensions=num_dimensions,
            init_scale=init_scale,
            min_scale=min_scale,
            tanh_mean=tanh_mean,
            fixed_scale=fixed_scale,
            w_init=w_init,
            b_init=b_init)
        self._use_tfd_independent = use_tfd_independent

    def __call__(self, inputs: tf.Tensor) -> tfd.Distribution:
        mean, scale = self._loc_scale(inputs)
        if self._use_tfd_independent:
            dist = tfd.Independent(tfd.Normal(loc=mean, scale=scale))
        else:
            dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=scale)

        return dist


class MultivariateNormalDiagLocScaleHead(snt.Module):
    """Module that produces mean and scale of a multivariate normal distribution."""

    def __init__(
            self,
            num_dimensions: int,
            init_scale: float = 0.3,
            min_scale: float = 1e-6,
            tanh_mean: bool = False,
            fixed_scale: bool = False,
            w_init: Optional[Initializer] = None,  # tf.initializers.VarianceScaling(1e-4),
            b_init: Optional[Initializer] = None  # tf.initializers.Zeros()
    ):
        """Initialization.

        Args:
          num_dimensions: Number of dimensions of MVN distribution.
          init_scale: Initial standard deviation.
          min_scale: Minimum standard deviation.
          tanh_mean: Whether to transform the mean (via tanh) before passing it to
            the distribution.
          fixed_scale: Whether to use a fixed variance.
          w_init: Initialization for linear layer weights.
          b_init: Initialization for linear layer biases.
        """
        super().__init__(name='MultivariateNormalDiagLocScaleHead')
        self._init_scale = init_scale
        self._min_scale = min_scale
        self._tanh_mean = tanh_mean
        self._mean_layer = snt.Linear(num_dimensions, w_init=w_init, b_init=b_init)
        self._fixed_scale = fixed_scale

        if not fixed_scale:
            self._scale_layer = snt.Linear(
                num_dimensions, w_init=w_init, b_init=b_init)

    def __call__(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        zero = tf.constant(0, dtype=inputs.dtype)
        mean = self._mean_layer(inputs)

        if self._fixed_scale:
            scale = tf.ones_like(mean) * self._init_scale
        else:
            scale = tf.nn.softplus(self._scale_layer(inputs))
            scale *= self._init_scale / tf.nn.softplus(zero)
            scale += self._min_scale

        # Maybe transform the mean.
        if self._tanh_mean:
            mean = tf.tanh(mean)

        return mean, scale
