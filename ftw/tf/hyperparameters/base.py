"""TensorFlow Hyperparameter API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Mapping, Any
import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Hyperparameter:
    """Metaclass that describes an API for hyperparameters.

    Has one private attribute:
        - _name: Name of the Hyperparameter.

    Offers the following public methods,
    which must be implemented by classes inheriting from Hyperparameter:
        -   get(): returns the value of the hyperparameter
        -   set(value): sets the value of the hyperparameter
        -   perturb(): perturbs the value of the hyperparameter.
    """

    _name: str = 'hyperparameter'

    @abc.abstractmethod
    def get(self):
        """Returns the value of the hyperparameter variable."""

    @abc.abstractmethod
    def set(self, value):
        """Sets the value of the hyperparameter variable."""

    @abc.abstractmethod
    def perturb(self):
        """Perturbs/Mutates the value of the hyperparameter variable."""

    def __repr__(self):
        return f"{self._name}: {self.get()}"


class HyperparametersContainer(Hyperparameter):
    # TODO: docstring

    def __init__(self, hypers_dict: Mapping[str, Hyperparameter]):
        self._hyperparameters = hypers_dict

    @property
    def hyperparameters(self):
        return self._hyperparameters

    def perturb(self):
        for hp in self._hyperparameters.values():
            hp.perturb()

    def get(self):
        return {name: hp.get() for name, hp in self._hyperparameters.items()}

    def set(self, value: Mapping[str, Any], verbose: bool = False):
        for v_name, v in value.items():
            if v_name in self._hyperparameters.keys():
                try:
                    self._hyperparameters[v_name].set(v)
                except Exception as e:
                    raise ValueError(
                        f"Could not set hyperparameter {v_name} to value {v} of type {type(v)}. "
                        f"Following exception occurred: {e}")
            elif verbose:
                print(f"Warning: Tried to set hyperparameter with name {v_name} and value {v} of type {type(v)} "
                      f"but this name does not exist in target HyperparametersContainer. Skipping...")
