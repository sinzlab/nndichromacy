from abc import ABC, abstractmethod
from mei.initial import InitialGuessCreator
import torch
from torch import Tensor, randn


def cumstom_initial_guess(*args, mean=0, std=1, device="cuda"):
    return torch.empty(*[args]).normal_(mean=mean, std=std).to(device)


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


class CustomRandomNormal(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = staticmethod(cumstom_initial_guess)

    def __init__(self, mean=111, std=60):
        self.mean = mean
        self.std = std

    def __call__(self, *shape, mean=111, std=60):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        return self._create_random_tensor(*shape, mean=self.mean, std=self.std)

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"
