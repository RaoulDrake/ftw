from typing import Union, Optional
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt


tfd = tfp.distributions
snt_init = snt.initializers
Initializer = Optional[Union[snt_init.Initializer, tf.initializers.Initializer]]

FloatValue = Union[float, np.float]
IntValue = Union[int, np.int]

FloatValueOrTFVariable = Union[float, tf.Variable]
IntValueOrTFVariable = Union[float, tf.Variable]

ArrayOrTensor = Union[np.ndarray, tf.Tensor]
