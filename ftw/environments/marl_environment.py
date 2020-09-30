"""Python MARL Environment API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc

import six


@six.add_metaclass(abc.ABCMeta)
class MARLEnvironment:
    """Abstract base class for Python competitive MARL environments.

    The main use-case assumes a competitive game with the following characteristics:
        -   The game has 2 teams.
        -   Both teams have the same number of players (1 vs 1 is also possible).
        -   The environment maps players to teams in a deterministic manner, ie. eg.
            in a 1 vs 1 environment, the first action of the joint action space vector supplied to
            environment.step() will always represent player 1 / team 1,
            while the second action will always be associated with player 2 / team 2.
            The recommended order of this mapping from joint action space indices to teams is to assign
            the first half of the joint action space vector to team 1 and the second half to team 2.
            This may seem like trivial commonsense but it is nonetheless important, since it ensures
            compatibility with population-based training methods that rely on ELO-values (or any similar
            statistics-based methods to judge a player's fitness in comparison to the rest of the population).
        -   Since it is a competitive environment, there are 3 possible outcomes for an episode:
            - 0.0 => Team 1 won / Team 2 lost.
            - 0.5 => Draw between both teams.
            - 1.0 => Team 2 won / Team 1 lost.
            Keeping the range of values between 0 and 1  ensures compatibility with
            ELO-based (or any similar statistics-based) population-based training methods.

    However, it can also be used to model a purely collaborative environment,
    in which case the following semantics should hold (in comparison/contrast to above):
        -   There is now only one team, consisting of one or more (collaborative) players.
        -   Possible outcomes for an episode are now in the range between 0 and 1, signifying the degree
            to which the team has successfully achieved its task.
    """

    @abc.abstractmethod
    def get_outcome(self):
        """Reports the outcome of an episode in the environment.

        Recommended possible return values in the main use-case of competitive environments are:
            0.0 => Team 1 won / Team 2 lost.
            0.5 => Draw between both teams.
            1.0 => Team 2 won / Team 1 lost.
        The recommended datatype of the return value is np.float32 (numpy).

        However, collaborative environments can also be modeled.
        In that case the above requirements can be ignored.
        """
