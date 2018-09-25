from abc import ABC, abstractmethod

import numpy as np

class BaseProcessor(ABC):
    """Abstract Processor Class

    Note
    ----
    You need to implement the follwoings:
    preprocess: preprocess observed data
    """
    def __call__(self, observation, action, reward, terminal):
        return self.preprocess(observation, action, reward, terminal)

    @abstractmethod
    def preprocess(self, observation, action, reward, terminal):
        raise NotImplementedError()


class Processor(BaseProcessor):
    def __init__(self, input_shape=None):
        self.input_shape = input_shape

    def preprocess(self, observation, action, reward, terminal):
        return observation, action, reward, terminal
