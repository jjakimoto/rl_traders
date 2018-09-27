from abc import ABC, abstractmethod

from tqdm import tqdm, tqdm_notebook

from ..processors import Processor


class BaseAgent(ABC):
    """Abstract Agentclass

    Parameters
    ----------
    action_spec: dict
        Have to define  'type' and 'shape'
    state_spec: dict, optional
        Have to define 'type' and 'shape'
    is_debug: bool
        If True, print out certain properties for debugging


    Note
    ----
    You need to define the followings:
        _observe: observe and store
    """

    def __init__(self, action_spec, state_spec=None,
                 processor=Processor,
                 is_debug=False, is_notebook=False, *args, **kwargs):
        self.action_spec = action_spec
        self.action_shape = self.action_spec["shape"]
        self.state_spec = state_spec
        self.processor = processor
        self.is_debug = is_debug
        if is_notebook:
            self.pbar = tqdm_notebook()
        else:
            self.pbar = tqdm()

    def observe(self, observation, action, reward, terminal, info, is_store):
        observation, action, reward, terminal =\
            self.processor(observation, action, reward, terminal)
        self.pbar.update(1)
        return self._observe(observation, action, reward, terminal, info, is_store)

    @abstractmethod
    def _observe(self, observation, action, reward, terminal, info, is_store):
        raise NotImplementedError("Need to define '_observe' at a subclass of Agent")

    def predict(self, observation, *args, **kwargs):
        action = self._predict(observation, *args, **kwargs)
        return action

    @abstractmethod
    def _predict(self, state, *args, **kwargs):
        raise NotImplementedError("Need to define '_predict' at a subclass of Agent")

    def fit(self, *args, **kwargs):
        pass

