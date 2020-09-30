from typing import Union, Tuple
from ftw.types import ArrayOrTensor
from ftw.tf import hyperparameters as hp

import numpy as np
from scipy import stats
import tensorflow as tf


class InternalRewards(hp.Hyperparameter):
    """Internal rewards class, as introduced by the FTW paper (Jaderberg et al., 2019).

    In addition to the implementation of Internal rewards as used in the FTW paper,
    this class also supports the 'base case' of a scalar internal reward variable.
    
    Can be initialized either with a concrete scalar value or a
    tuple (min, max) indicating a range from which to draw a random sample.
    More specifically, the sample is drawn from a log-uniform distribution
    defined over this range.

    This class is used in combination with environments that offer a vector
    of different environment events, which should be supplied by the environment
    instead of a normal scalar reward, e.g. as the reward field of a
    dm-acme observation_action_reward.OAR NamedTuple.
    This class offers a reward() method that computes reward as a
        -   dot product between environment events and internal reward weights,
            if internal rewards is a vector and events is a (batch of) vector(s).
        -   product between environment events and internal reward weights,
            if internal rewards is a scalar and events is a (batch of) vector(s).
    In the unlikely case that events is a scalar, but internal rewards a vector,
    reward() will raise a ValueError.

    Inherits from ftw.tf.hyperparameters.Hyperparameter,
    i.e. it offers get(), set() and perturb() methods.
    See docstring for ftw.tf.hyperparameters.Hyperparameter for more details.
    """

    def __init__(self,
                 num_events: int,
                 init_value_or_range: Union[float, Tuple[float, float]],
                 perturb_probability: float = 1.0,  # 0.05,
                 perturb_max_pct_change: float = 0.20,
                 dtype=tf.float32,
                 name: str = 'internal_rewards'):
        """Initializes InternalRewards.

        Args:
            num_events: Number of different events supplied by the environment.
                Must be an int value > 0.
            init_value_or_range: A scalar value of type int for initialization with an exact value,
                or a tuple (min, max), where min, max are of type int for initialization by drawing
                a sample from a log-uniform distribution defined over (min, max).
            perturb_probability: The probability of actually perturbing the
                internal rewards value(s) when calling perturb().
                Must be a float value with 0 <= value <= 1.
            perturb_max_pct_change: Maximum allowed change of the hyperparameter value in percent,
                when calling perturb() and if perturb() actually perturbs the value
                (see perturb_probability). Resulting change lies in the range of
                (-perturb_max_pct_change, perturb_max_pct_change).
                Must be a float value > 0.
            dtype: tf.dtype used by the tf.Variable that holds the internal rewards value(s).
                Defaults to tf.float32.
            name: Name for this InternalRewards instance. Defaults to 'internal_rewards'.
        """
        if not (isinstance(num_events, int) or isinstance(num_events, np.int)):
            raise ValueError("num_events must be of type int.")
        if num_events < 1:
            raise ValueError("num_events must be > 0")
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
            raise ValueError(f"perturb_probability must be of type float "
                             f"but was of type {type(perturb_probability)}.")
        if not isinstance(perturb_max_pct_change, float):
            raise ValueError(f"perturb_max_pct_change must be of type float "
                             f"but was of type {type(perturb_max_pct_change)}.")
        if perturb_probability < 0.0 or perturb_probability > 1.0:
            raise ValueError(f"perturb_probability must be >= 0 and <= 1 but was {perturb_probability}.")
        if perturb_max_pct_change < 0.0:
            raise ValueError(f"perturb_max_pct_change must be >= 0 but was {perturb_max_pct_change}.")        
        
        if isinstance(init_value_or_range, float):
            # Ensure initial_value and num_events are compatible for tf.Variable initialization.
            initial_value = [init_value_or_range] * num_events
        else:
            # Initialize with internal reward weights drawn from a log-uniform distribution.
            initial_value = stats.loguniform.rvs(init_value_or_range[0], init_value_or_range[1], size=num_events)

        self._internal_rewards = tf.Variable(
            initial_value=initial_value,
            trainable=False,
            dtype=dtype,
            shape=[num_events],
            name=name)
        self._perturb_probability = perturb_probability
        self._perturb_max_pct_change = perturb_max_pct_change
        self._name = name
        self._num_events = num_events

    def get(self) -> np.ndarray:
        """Returns (numpy) value of the tf.Variable storing the hyperparameter value."""
        return self._internal_rewards.numpy()

    def set(self, value: ArrayOrTensor):
        """Assign value to the tf.Variable storing the hyperparameter value.

        Args:
            value: numpy.ndarray or tf.Tensor, containing the new value for the hyperparameter variable.
        """
        self._internal_rewards.assign(value)

    @property
    def variable(self) -> tf.Variable:
        """Returns the tf.Variable storing the hyperparameter value."""
        return self._internal_rewards

    def perturb(self, verbose=False):
        """May perturb the internal rewards, depending on perturb_probability."""
        if self._perturb_probability == 1.0 or np.random.random() < self._perturb_probability:
            pct_change = np.random.uniform(low=-self._perturb_max_pct_change,
                                           high=self._perturb_max_pct_change,
                                           size=self._num_events)
            new_value = self.get() + (pct_change * self.get())
            if verbose:
                print(f"{self._name}\tPct Change: {pct_change}\tNew Value: {new_value}")
            self.set(new_value)

    def reward(self, events: ArrayOrTensor) -> tf.Tensor:
        """Computes reward as a dot product between environment events and internal reward weights.

        If internal reward is a scalar, then the reward is just the
        product between events and internal reward.

        Args:
            events: Environment events. Expected to be of type numpy.ndarray or tf.Tensor.

        Returns:
            Scalar reward of type tf.Tensor.

        Raises:
            ValueError: If shapes of events and internal rewards are incompatible.
        """
        if self._num_events == 1:  # Scalar internal rewards.
            return events * self._internal_rewards
        elif events.shape != ():  # Non-scalar events.
            if events.shape[-1] == self._num_events:
                # Number of non-scalar internal rewards matches the number of events
                # (i.e. the last dimension of possibly batched events).
                # Compute the dot product between events and internal rewards.
                return tf.tensordot(events, tf.stop_gradient(self._internal_rewards), axes=1)
            else:
                raise ValueError('Last dimension of events must equal num_events passed to '
                                 'InternalRewards constructor.')
        else:
            raise ValueError('Scalar events are not compatible with non-scalar internal rewards.')
