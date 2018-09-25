import numpy as np
from collections import namedtuple
from copy import deepcopy

from .core import BaseMemory
from .utils import RingBuffer, sample_batch_indexes

Experience = namedtuple('Experience',
                        'state, previous_action, reward, terminal, index')


class PortfolioVectorMemory(BaseMemory):
    """Memory with portfolio vector

    Parameters
    ----------
    limit: int
        The maximum size of the memory
    window_length: int
        The length of trailing history for input
    beta: float, (default 5.0e-5)
        Prioritization parameter for sampling
    """

    def __init__(self, limit, window_length, beta=5.0e-5,
                 keys=['close', 'high', 'low', 'volume'],
                 *args, **kwargs):
        super(PortfolioVectorMemory, self).__init__(
            window_length=window_length,
            keys=keys, *args, **kwargs)
        self.limit = limit
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = dict()
        for key in self.keys:
            self.observations[key] = RingBuffer(limit)
        self.recent_action = None
        self.beta = beta

    def sample(self, batch_size, newest=False, *args, **kwargs):
        """Sample batches with time prioritization

        Parameters
        ----------
        batch_size: int
            The size of batch
        newest: bool, (defaualt False)
            If True, use newset data for sampling without randomness

        Returns
        -------
        list of experiences
        """
        if newest:
            batch_idxs = np.arange(self.nb_entries - batch_size,
                                   self.nb_entries)
        else:
            num_candidates = self.nb_entries - (
                        batch_size + self.window_length - 2)
            # Index has to be more than one at least for taking prev_action
            num_candidates -= 1
            # You need to have at least one cadidate
            assert num_candidates > 0
            idx = np.random.geometric(self.beta) % num_candidates
            init_idx = self.nb_entries - batch_size - idx
            batch_idxs = np.arange(init_idx, init_idx + batch_size)
        # Create experiences
        experiences = []
        # Each idx is index for state1
        for i, idx in enumerate(batch_idxs):
            # Order: observation => action => (reward, terminal, info)
            # previous index has to be terminal==False.
            previous_action = self.actions[idx - 1]
            reward = self.rewards[idx]
            terminal = self.terminals[idx]
            state = self.get_state(idx)
            experiences.append(Experience(state=state,
                                          previous_action=previous_action,
                                          reward=reward, terminal=terminal,
                                          index=idx))
        assert len(experiences) == batch_size
        return experiences

    def _append(self, observation, action, reward, terminal, info,
                is_store=True):
        self.recent_action = deepcopy(action)
        if is_store:
            for key in self.keys:
                self.observations[key].append(observation[key])
            self.actions.append(action)
            self.rewards.append(info["returns"])
            self.terminals.append(terminal)

    def _sample_batch_indexes(self, low, high, size, weights=None):
        return sample_batch_indexes(low, high, size, weights)

    @property
    def nb_entries(self):
        return len(self.actions)

    @property
    def start_idx(self):
        return self.nb_entries - 1

    def get_recent_action(self):
        return self.recent_action

    def get_recent_state(self, current_observation):
        # each recent state has shape=(window_length, n_currencies)
        closes = self._get_recent_state(current_observation, 'close')
        highs = self._get_recent_state(current_observation, 'high')
        lows = self._get_recent_state(current_observation, 'low')
        volumes = self._get_recent_state(current_observation, 'volume')
        return self._get_normalized_state(closes, highs, lows, volumes)

    def get_state(self, idx, state=None):
        """Return list of last observations


        Parameters
        ----------
        idx: int
            The index of the most recent observation in state
        state: list, optional

        Returns
        -------
        A list of observations
        """
        state_list = []
        terminals = self.terminals
        closes = self._get_state(idx, terminals, self.observations['close'],
                                 state)
        highs = self._get_state(idx, terminals, self.observations['high'],
                                state)
        lows = self._get_state(idx, terminals, self.observations['low'], state)
        volumes = self._get_state(idx, terminals, self.observations['volume'],
                                  state)
        return self._get_normalized_state(closes, highs, lows, volumes)

    def update_portfolio_vector(self, action, idx):
        """Update action memory for Portfolio Vector Memory"""
        self.actions[idx] = action

    def _get_normalized_state(self, closes, highs, lows, volumes):
        # Normalize price
        price_scale = closes[-1]
        highs /= price_scale
        lows /= price_scale
        closes /= price_scale
        # Normalize volume (Volume could be zero)
        min_vol = np.min(volumes, axis=0)
        max_vol = np.max(volumes, axis=0)
        volumes = (volumes - min_vol) / (max_vol - min_vol)
        # shape=(4, window_length, n_currencies)
        state = np.stack([closes, highs, lows, volumes], axis=0)
        return state