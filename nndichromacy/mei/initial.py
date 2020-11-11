from abc import ABC, abstractmethod
from mei.initial import InitialGuessCreator
from torch import Tensor, randn
from .utility import cumstom_initial_guess
from functools import partial

class RandomNormalBehavior(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __init__(self, first_behav_channel, channel_0, channel_1, channel_2):
        self.first_behav_channel = first_behav_channel
        self.channel_0 = channel_0
        self.channel_1 = channel_1
        self.channel_2 = channel_2

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        inital = self._create_random_tensor(*shape)
        inital[:, self.first_behav_channel, ...] = self.channel_0
        inital[:, self.first_behav_channel+1, ...] = self.channel_1
        inital[:, self.first_behav_channel+2, ...] = self.channel_2
        return inital

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


eightbit_initial_guess = partial(cumstom_initial_guess, mean=111, std=60)
