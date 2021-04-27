from abc import ABC, abstractmethod
from mei.initial import InitialGuessCreator
import torch
from torch import Tensor, randn
from collections import Iterable
from dataport.bcm.color_mei.schema import StaticImage
from ..tables.utils import preprocess_img_for_reconstruction
import numpy as np


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


class NaturalImgBehavior(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    def __init__(self, image_class, image_id, first_behav_channel, channel_0, channel_1, channel_2, img_size=(36, 64)):

        image = (StaticImage.Image & dict(image_class=image_class, image_id=image_id)).fetch1("image")
        image = preprocess_img_for_reconstruction(image, img_size=img_size)
        # standardization
        image = (image-image.mean()) / image.std()
        self.natural_img = image
        self.first_behav_channel = first_behav_channel
        self.channel_0 = channel_0
        self.channel_1 = channel_1
        self.channel_2 = channel_2

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        initial = torch.ones(*shape).cuda()
        n_channels_img = self.natural_img.shape[1]
        initial[:, :n_channels_img, ...] = self.natural_img
        initial[:, self.first_behav_channel, ...] = self.channel_0
        initial[:, self.first_behav_channel + 1, ...] = self.channel_1
        initial[:, self.first_behav_channel + 2, ...] = self.channel_2
        return initial

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class GrayNormalBehavior(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    def __init__(self, first_behav_channel, channel_0, channel_1, channel_2):
        self.first_behav_channel = first_behav_channel
        self.channel_0 = channel_0
        self.channel_1 = channel_1
        self.channel_2 = channel_2

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        inital = torch.zeros(*shape)
        inital[:, self.first_behav_channel, ...] = self.channel_0
        inital[:, self.first_behav_channel+1, ...] = self.channel_1
        inital[:, self.first_behav_channel+2, ...] = self.channel_2
        return inital

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class RandomNormalSelectedChannels(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __init__(self, selected_channels, selected_values):
        if not isinstance(selected_channels, Iterable):
            selected_channels = (selected_channels)

        if not isinstance(selected_values, Iterable):
            selected_values = (selected_values)

        self.selected_channels = selected_channels
        self.selected_values = selected_values

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        inital = self._create_random_tensor(*shape)
        for channel, value in zip(self.selected_channels, self.selected_values):
            inital[:, channel, ...] = value

        return inital

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class RandomNormBehaviorPositions(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __init__(self, selected_channels, selected_values):
        if not isinstance(selected_channels, Iterable) and (selected_channels is not None):
            selected_channels = (selected_channels)

        if not isinstance(selected_values, Iterable) and (selected_values is not None):
            selected_values = (selected_values)

        self.selected_channels = selected_channels
        self.selected_values = selected_values

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""

        inital = self._create_random_tensor(*shape)
        if self.selected_channels is not None:
            for channel, value in zip(self.selected_channels, self.selected_values):
                inital[:, channel, ...] = value
        inital[:, -2:, ...] = torch.from_numpy(np.stack(np.meshgrid(np.linspace(-1, 1, shape[-1]), np.linspace(-1, 1, shape[-2]))))
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
