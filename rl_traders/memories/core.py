from collections import deque, namedtuple
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from .utils import zeroed_observation


class BaseMemory(ABC):
    """Base class for memory

    Note
    ----
    You need to implement the followings:
        _append: add new observations to memory

    Parameters
    ----------
    window_length: int
        The length to be used for input
    ignore_episode_boundaries: bool
        If False, terminal is used without considering the boundary
        when sampling or getting recdent state
    """
    def __init__(self, window_length, keys=['close', 'high', 'low', 'volume'],
                 ignore_episode_boundaries=False):
        self.window_length = window_length
        self.keys = keys
        self.ignore_episode_boundaries = ignore_episode_boundaries
        self.recent_observations = dict()
        for key in keys:
            self.recent_observations[key] = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)

    def sample(self, batch_size, batch_idxs=None):
        raise NotImplementedError()

    def append(self, observation, action, reward, terminal, info, is_store=True):
        # We do not store the final state
        for key in self.keys:
            self.recent_observations[key].append(observation[key])
        self.recent_terminals.append(terminal)
        self._append(observation, action, reward, terminal, info, is_store)

    def _get_state(self, idx, terminals=None, observations=None, state=None):
        """Return list of last observations


        Parameters
        ----------
        idx: int
            The index of the most recent observation in state
        terminals: list, optional
        observations: list, optional
        state: list, optional

        Returns
        -------
        A list of observations
        """
        # This code is slightly complicated by the fact that subsequent observations might be
        # from different episodes. We ensure that an experience never spans multiple episodes.
        # This is probably not that important in practice but it seems cleaner.
        if state is None:
            state = []
        if observations is None:
            observations = self.observations
        if terminals is None:
            terminals = self.terminals
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            # Order: observation => action => (reward, terminal, info)
            if current_idx >= 0:
                current_terminal = terminals[current_idx]
            else:
                break
            if not self.ignore_episode_boundaries and current_terminal:
                # The previously handled observation was terminal, don't add the current one.
                # Otherwise we would leak into a different episode.
                break
            state.insert(0, observations[current_idx])
        while len(state) < self.window_length:
            state.insert(0, zeroed_observation(state[0]))
        state = np.stack(state, 0)
        return np.array(state)

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
        for key in self.keys:
            observations = self.observations[key]
            state_list.append(self._get_state(idx, terminals,
                                              observations, state))
        return np.stack(state_list, axis=1)

    def _get_recent_state(self, current_observation, obs_key):
        """Return list of last observations


        Parameters
        ----------
        current_observation: dict(array-like)
            The lastest observation
        obs_key: str
            The key for observation to be used

        Returns
        -------
        A list of the last observations
        """
        terminals = self.recent_terminals
        observations = self.recent_observations[obs_key]
        state = [current_observation[obs_key]]
        idx = len(observations) - 1
        return self._get_state(idx, terminals, observations, state=state)

    def get_recent_state(self, current_observation):
        """Return list of last observations


        Parameters
        ----------
        current_observation: dict(array-like)
            The lastest observation

        Returns
        -------
        A list of the last observations
        """
        state_list = []
        for key in self.keys:
            curr_obs = deepcopy(current_observation)
            state_list.append(self._get_recent_state(curr_obs, key))
        return np.stack(state_list, axis=1)

    def get_config(self):
        """Return configuration (window_length, ignore_episode_boundaries) for Memory

        # Return
            A dict with keys window_length and ignore_episode_boundaries
        """
        config = {
            'window_length': self.window_length,
            'ignore_episode_boundaries': self.ignore_episode_boundaries,
        }
        return config

    @abstractmethod
    def _append(self, observation, action, reward, terminal, training):
        raise NotImplementedError()

