from typing import Union, List, Mapping

import collections

from acme import core

from ftw.tf import hyperparameters as hp
from ftw.tf import internal_reward
from ftw.agents.tf import ftw as ftw_agent
from ftw.pbt.jobs import Job

import numpy as np
from scipy import stats


def normal_pdf(x):
    return stats.norm.pdf(x, loc=0.5, scale=1/6)


class Chief(Job):

    def __init__(self, players, team_size, min_steps=10000000, T=200):
        self._elos = np.ones(players) * 1000
        self._team_size = team_size
        self._players = players
        self._min_steps = min_steps
        self._last_pbtd = np.zeros(players)
        self._T = T

    def _update_elo(self, players, outcome):
        # TODO
        pass

    def get_match(self):
        # Select first agent at random
        players = [np.random.choice(self._players)]

        # Create a pool of potential teammates and opponents
        player_pool = set(range(self._players))
        player_pool.remove(players[0])

        # Fill in the remaining slots
        for _ in range(self._team_size * 2 - 1):
            # Sample proportionally to strength similarity
            weights = [normal_pdf(self._elos[players[0]] - self._elos[player])
                       for player in sorted(player_pool)]
            weights = np.array(weights)
            weights /= weights.sum()

            player = np.random.choice(list(sorted(player_pool)), p=weights)
            players.append(player)

            # Sample without replacement
            player_pool.remove(player)

        return players

    def process_result(self, players, outcome):
        self._update_elo(players, outcome)

    def run(self):
        while True:
            for agent_id, learner in enumerate(self._job_pool.learners):
                # Check if agent has not evolved for a long time
                if learner.get_step() > self._last_pbtd[agent_id] + self._min_steps:

                    # Find candidate replacement at random
                    candidate = np.random.choice(range(self._players))

                    # Check if candidate is stronger
                    if self._elos[candidate] > self._elos[agent_id] + self._T:
                        parent = self._job_pool.learners[candidate]

                        # Copy weights
                        learner.set_weights(parent.get_weights())

                        # Copy & perturb reward scheme
                        new_rewards = parent.get_rewards()
                        if np.random.random() < 0.1:
                            new_rewards = new_rewards.perturb()
                        learner.set_rewards(new_rewards)

                        # Copy & perturb hyperparameters
                        new_hypers = parent.get_hypers()
                        if np.random.random() < 0.1:
                            new_hypers = new_hypers.perturb()
                        learner.set_hypers(new_hypers)

                        # Record evolution time
                        learner.set_steps(parent.get_step())
                        self._last_pbtd[agent_id] = learner.get_step()
