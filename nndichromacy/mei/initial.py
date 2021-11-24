from abc import ABC, abstractmethod
from mei.initial import InitialGuessCreator
import torch
from torch import Tensor, randn
from collections import Iterable
from dataport.bcm.color_mei.schema import StaticImage
from ..tables.utils import preprocess_img_for_reconstruction
import numpy as np
from ..tables.from_mei import MEI
import os

fetch_download_path = os.environ.get('FETCH_DOWNLOAD_PATH', '/data/fetched_from_attach')


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
        inital[:, self.first_behav_channel + 1, ...] = self.channel_1
        inital[:, self.first_behav_channel + 2, ...] = self.channel_2
        return inital

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class NaturalImgBehavior(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    def __init__(
        self,
        image_class,
        image_id,
        first_behav_channel,
        channel_0,
        channel_1,
        channel_2,
        img_size=(36, 64),
    ):

        image = (
            StaticImage.Image & dict(image_class=image_class, image_id=image_id)
        ).fetch1("image")
        image = preprocess_img_for_reconstruction(image, img_size=img_size)
        # standardization
        image = (image - image.mean()) / image.std()
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
        inital[:, self.first_behav_channel + 1, ...] = self.channel_1
        inital[:, self.first_behav_channel + 2, ...] = self.channel_2
        return inital

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class RandomNormalSelectedChannels(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __init__(self, selected_channels, selected_values):
        if not isinstance(selected_channels, Iterable):
            selected_channels = selected_channels

        if not isinstance(selected_values, Iterable):
            selected_values = selected_values

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
        if not isinstance(selected_channels, Iterable) and (
            selected_channels is not None
        ):
            selected_channels = selected_channels

        if not isinstance(selected_values, Iterable) and (selected_values is not None):
            selected_values = selected_values

        self.selected_channels = selected_channels
        self.selected_values = selected_values

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""

        inital = self._create_random_tensor(*shape)
        if self.selected_channels is not None:
            for channel, value in zip(self.selected_channels, self.selected_values):
                inital[:, channel, ...] = (
                    value.item()
                    if not (isinstance(value, float) or isinstance(value, int))
                    else value
                )
        inital[:, -2:, ...] = torch.from_numpy(
            np.stack(
                np.meshgrid(
                    np.linspace(-1, 1, shape[-1]), np.linspace(-1, 1, shape[-2])
                )
            )
        )
        return inital

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class RandomNormBehaviorPositionsAdaptivePupil(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __init__(self, selected_channels, selected_values, key, threshold_percentile):
        if not isinstance(selected_channels, Iterable) and (
            selected_channels is not None
        ):
            selected_channels = selected_channels

        if not isinstance(selected_values, Iterable) and (selected_values is not None):
            selected_values = selected_values

        self.selected_channels = selected_channels
        self.selected_values = selected_values

        from ..tables.from_nnfabrik import Dataset

        dataloaders = (Dataset & key).get_dataloader()
        behaviors = []
        for b in dataloaders["train"][key["data_key"]]:
            behaviors.append(b.behavior.cpu().numpy())
        behaviors = np.vstack(behaviors)

        pupil_threshold = np.percentile(behaviors[:, 0], threshold_percentile)
        pupil = behaviors[:, 0].max()
        while pupil > pupil_threshold:
            randint = np.random.choice(range(behaviors.shape[0]))
            pupil = behaviors[randint, 0]
            running = behaviors[randint, 2]

        self.selected_values[0] = pupil
        self.selected_values[2] = running
        print("done. Selected Behaviors: ", pupil, running)

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""

        inital = self._create_random_tensor(*shape)
        if self.selected_channels is not None:
            for channel, value in zip(self.selected_channels, self.selected_values):
                inital[:, channel, ...] = (
                    value.item()
                    if not (isinstance(value, float) or isinstance(value, int))
                    else value
                )
        inital[:, -2:, ...] = torch.from_numpy(
            np.stack(
                np.meshgrid(
                    np.linspace(-1, 1, shape[-1]), np.linspace(-1, 1, shape[-2])
                )
            )
        )
        return inital

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"

class RandomNormBehaviorPositionsSurr(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __init__(self, selected_channels, selected_values, key, mask_thres=0.3):
        if not isinstance(selected_channels, Iterable) and (selected_channels is not None):
            selected_channels = (selected_channels)

        if not isinstance(selected_values, Iterable) and (selected_values is not None):
            selected_values = (selected_values)

        self.selected_channels = selected_channels
        self.selected_values = selected_values
        
        src_method_fn = key["src_method_fn"]
        unit_id = key["unit_id"]

        inner_ensemble_hash = key["inner_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        unit_id = key["unit_id"]

        inner_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=inner_ensemble_hash) & dict(method_hash=inner_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        
        #outer_mei=torch.load(outer_mei_path)
        inner_mei=torch.load(inner_mei_path)

        self.center_mask= (inner_mei[0][-1] > mask_thres) * 1
        #self.center_type = center_type
        #self.center_norm = center_norm
        self.centerimg = inner_mei[0][:2]
        
    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""

        initial = self._create_random_tensor(*shape)
        if self.selected_channels is not None:
            for channel, value in zip(self.selected_channels, self.selected_values):
                initial[:, channel, ...] = value
        initial[:, -2:, ...] = torch.from_numpy(np.stack(np.meshgrid(np.linspace(-1, 1, shape[-1]), np.linspace(-1, 1, shape[-2]))))
        initial[:,:2, ...] = self.centerimg + initial[:,:2,...] * (1-self.center_mask)
        return initial

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
