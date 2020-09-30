from typing import Union, Tuple, Optional
from ftw.types import FloatValue, IntValue
import numpy as np
from scipy import stats
import tensorflow as tf
from ftw.tf.hyperparameters import base


class FloatHyperparameter(base.Hyperparameter):
    """Hyperparameter with a scalar float value.

    Can be initialized either with a concrete scalar value or a
    tuple (min, max) indicating a range from which to draw a random sample.
    More specifically, the sample is drawn from a log-uniform distribution
    defined over this range.

    Contains a tf.Variable that stores the hyperparameter value. Consequently,
    this module is well suited to be used as a hyperparameter variable inside a tf.function, e.g.
    in population-based training.
    """

    def __init__(self,
                 init_value_or_range: Union[float, Tuple[float, float]],
                 perturb_probability: float = 1.0,
                 perturb_max_pct_change: float = 0.20,
                 dtype=tf.float32,
                 name: str = 'float_hyperparameter'):
        """
        Args:
            init_value_or_range: A scalar value of type int for initialization with an exact value,
                or a tuple (min, max), where min, max are of type int for initialization by drawing
                a sample from a log-uniform distribution defined over (min, max).
            perturb_probability: The probability of actually perturbing the hyperparameter value
                when calling perturb(). Must be a float value with 0 <= value <= 1..
            perturb_max_pct_change: Maximum allowed change of the hyperparameter value in percent,
                when calling perturb() and if perturb() actually perturbs the value
                (see perturb_probability). Resulting change lies in the range of
                (-perturb_max_pct_change, perturb_max_pct_change).
                Must be a float value > 0.
            dtype: tf.dtype used by the tf.Variable that holds the hyperparameter value.
                Defaults to tf.float32.
            name: Name for this FloatHyperparameter instance. Defaults to 'float_hyperparameter'.
        """
        if not (
                isinstance(init_value_or_range, float) or
                (isinstance(init_value_or_range, tuple) and
                 all([isinstance(x, float) for x in init_value_or_range]))
        ):
            raise ValueError(f"init_value_or_range must be of type float, "
                             f"to initialize the hyperparameter with an exact value, or a tuple (min, max), "
                             f"to draw the initialization value from a log-uniform distribution, "
                             f"where min and max must also be of type float."
                             f"init_value_or_range supplied: {init_value_or_range}")
        if not isinstance(perturb_probability, float):
            raise ValueError("perturb_probability must be of type float.")
        if not isinstance(perturb_max_pct_change, float):
            raise ValueError("perturb_max_pct_change must be of type float.")
        if perturb_probability < 0.0 or perturb_probability > 1.0:
            raise ValueError(f"perturb_probability must be >= 0 and <= 1 but was {perturb_probability}.")
        if perturb_max_pct_change < 0.0:
            raise ValueError(f"perturb_max_pct_change must be >= 0 but was {perturb_max_pct_change}.")

        if isinstance(init_value_or_range, float):
            initial_value = init_value_or_range
        else:
            initial_value = stats.loguniform.rvs(init_value_or_range[0], init_value_or_range[1])

        self._variable = tf.Variable(
            initial_value=initial_value,
            trainable=False,
            dtype=dtype,
            shape=())
        self._perturb_probability = perturb_probability
        self._perturb_max_pct_change = perturb_max_pct_change
        self._name = name

    @property
    def variable(self):
        """Returns the tf.Variable that holds the hyperparameter value."""
        return self._variable

    def get(self):
        """Returns the value of the hyperparameter variable."""
        return self._variable.numpy()

    def set(self, value: FloatValue):
        """Sets the value of the hyperparameter variable.

        Args:
            value: Value of type float or numpy float.
        """
        self._variable.assign(value)

    def perturb(self, verbose=False):
        """Perturbs/mutates the value of the hyperparameter variable with perturb_probability."""
        if self._perturb_probability == 1.0 or np.random.random() < self._perturb_probability:
            pct_change = np.random.uniform(low=-self._perturb_max_pct_change,
                                           high=self._perturb_max_pct_change)
            new_value = self.get() + (pct_change * self.get())
            if verbose:
                print(f"{self._name}\tPct Change: {pct_change}\tNew Value: {new_value}")
            self.set(new_value)


class IntHyperparameter(base.Hyperparameter):
    """Hyperparameter with an int value.

    Supported initialization and perturbation behaviour of hyperparameter variable:
        -   Supplying min_value and max_value parameters, but no initial_value parameter results in:
            Initialization by drawing a random sample from a categorical distribution
            defined over the range [min_value, ..., max_value].
            Perturbation by drawing from the same distribution.
        -   Supplying min_value, max_value and initial_value parameters results in:
            Initialization with initial_value.
            Perturbation by drawing a random sample from a categorical distribution
            defined over the range [min_value, ..., max_value].

    Contains a tf.Variable that stores the hyperparameter value. Consequently,
    this module is well suited to be used as a hyperparameter variable inside a tf.function, e.g.
    in population-based training.
    """

    def __init__(self,
                 min_value: int,
                 max_value: int,
                 initial_value: Optional[int] = None,
                 perturb_probability: float = 1.0,  # 0.05,
                 dtype=tf.float32,
                 name: str = 'int_hyperparameter'):
        """
        Args:
            min_value: Inclusive lower limit of the range used for random initialization (if no init_value is supplied)
                and perturbation sampling. Python int or numpy int expected.
            max_value: Inclusive upper limit of the range used for random initialization (if no init_value is supplied)
                and perturbation sampling. Python int or numpy int expected.
            initial_value: Optional. If given, hyperparameter will be initialized with this value
                instead of being randomly initialized from the range defined by (min_value, max_value).
                Python int or numpy int expected.
            perturb_probability: The probability of actually perturbing the hyperparameter value
                when calling perturb(). Must be a float value with 0 <= value <= 1.
            dtype: tf.dtype used by the tf.Variable that holds the hyperparameter value.
                Defaults to tf.float32, even though the hyperparameter is an int,
                since this is often more compatible with common neural network models.
            name: Name for this IntHyperparameter instance. Defaults to 'int_hyperparameter'.

        Raises:
            ValueError: If invalid constructor arguments were passed.
        """
        if not isinstance(min_value, int):
            raise ValueError("min_value must be of type int.")
        if not isinstance(max_value, int):
            raise ValueError("max_value must be of type int.")
        if (initial_value is not None) and not(isinstance(initial_value, int)):
            raise ValueError(f"initial_value must be of type int. "
                             f"initial_value supplied: {initial_value}")
        if not isinstance(perturb_probability, float):
            raise ValueError("perturb_probability must be of type float.")
        if perturb_probability < 0.0 or perturb_probability > 1.0:
            raise ValueError(f"perturb_probability must be >= 0 and <= 1 but was {perturb_probability}.")

        self._min_value = min_value
        self._max_value = max_value
        if initial_value is None:
            initial_value = np.random.choice(
                np.array(list(range(min_value, max_value+1))))

        self._variable = tf.Variable(
            initial_value=initial_value,
            trainable=False,
            dtype=dtype,
            shape=(),
            name=name)

        self._perturb_probability = perturb_probability
        self._name = name

    @property
    def variable(self):
        """Returns the tf.Variable that holds the hyperparameter value."""
        return self._variable

    def get(self):
        """Returns the value of the hyperparameter variable."""
        return self._variable.numpy()

    def set(self, value: IntValue):
        """Sets the value of the hyperparameter variable.

        Args:
            value: Value of type int or numpy.int.
        """
        self._variable.assign(value)

    def perturb(self):
        """Perturbs/mutates the value of the hyperparameter variable with perturb_probability."""
        if self._perturb_probability == 1.0 or np.random.random() < self._perturb_probability:
            new_value = np.random.choice(
                np.array(list(range(self._min_value, self._max_value+1))))
            self.set(new_value)
