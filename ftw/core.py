import abc
from acme import core as acme_core


class LearnerStandalone(abc.ABC):
    """An interface for Learners that offer a method to create new actors."""

    def make_actor(self) -> acme_core.Actor:
        """Creates new adder(s) and a new actor and returns the actor"""
