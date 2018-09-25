from six.moves import xrange
from collections import deque
import random
import warnings
import numpy as np


def sample_batch_indexes(low, high, size):
    """Return a sample of (size) unique elements between low and high

    Parameters
    ----------
    low: int, The minimum value for our samples
    high: int, The maximum value for our samples
    size: int, The number of samples to pick

    Returns
    -------
    A list of samples of length size, with values between low and high
    """
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.
        r = xrange(low, high)
        batch_idxs = random.sample(r, size)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        warnings.warn(
            'Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs


class RingBuffer(object):
    """Erase the oldest memory after reaching maxlen

    Parameters
    ----------
    maxlen: int
        The maximum number of elements in memory
    """
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.data = deque(maxlen=maxlen)

    def __len__(self):
        return self.length()

    def __getitem__(self, idx):
        """Return element of buffer at specific index"""
        if idx < 0 or idx >= self.length():
            raise KeyError()
        return self.data[idx]

    def __setitem__(self, idx, value):
        """Set element by accessing with index"""
        if idx < 0 or idx >= self.length():
            raise KeyError()
        self.data[idx] = value

    def append(self, v):
        """Append an element to the buffer
        # Argument
            v (object): Element to append
        """
        self.data.append(v)

    def length(self):
        """Return the length of Deque
        # Argument
            None
        # Returns
            The lenght of deque element
        """
        return len(self.data)


def zeroed_observation(observation):
    """Return an array of zeros with same shape as given observation

    Returns
    -------
    observation: np.ndarray, list, or something else

    Returns
    -------
    A np.ndarray of zeros with observation.shape
    """
    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape)
    elif hasattr(observation, '__iter__'):
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.
