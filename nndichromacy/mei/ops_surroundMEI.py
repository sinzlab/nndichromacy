import warnings
import numpy as np

from abc import ABC, abstractmethod
import torch
from torch import Tensor, randn
from nnfabrik import builder

import torch.nn.functional as F
from scipy import signal
from collections.abc import Iterable
from mei.initial import InitialGuessCreator
from mei.legacy.utils import varargin
from ..tables.scores import MEINorm, MEINormBlue, MEINormGreen
from ..tables.from_mei import MEI
from ..tables.mei_scores import MEIThresholdMask
import datajoint as dj

import copy
import os
fetch_download_path = os.environ.get('FETCH_DOWNLOAD_PATH', '/data/fetched_from_attach')

class RandomNormBehaviorPositionsSurr(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __init__(self, selected_channels, selected_values, key):
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
        inner_mask_hash = key['mask_hash']

        mei_key = dj.AndList([dict(method_fn=src_method_fn),
                     dict(ensemble_hash=inner_ensemble_hash),
                     dict(method_hash=inner_method_hash),
                     dict(unit_id=unit_id)])

        inner_mei_path = (MEI & mei_key).fetch1('mei', download_path=fetch_download_path)
        inner_mei=torch.load(inner_mei_path)
        self.centerimg = inner_mei[0][:2]

        center_mask= (MEIThresholdMask & mei_key & dict(mask_hash=inner_mask_hash)).fetch1( "mask", download_path=fetch_download_path)
        self.center_mask=torch.tensor(center_mask)
        
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

class RandomInitialBehaviorSurr(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __init__(self, selected_channels, key):
        if not isinstance(selected_channels, Iterable) and (selected_channels is not None):
            selected_channels = (selected_channels)

        self.selected_channels = selected_channels
        
        src_method_fn = key["src_method_fn"]
        unit_id = key["unit_id"]
        inner_ensemble_hash = key["inner_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        inner_mask_hash = key['mask_hash']

        mei_key = dj.AndList([dict(method_fn=src_method_fn),
                     dict(ensemble_hash=inner_ensemble_hash),
                     dict(method_hash=inner_method_hash),
                     dict(unit_id=unit_id)])

        inner_mei_path = (MEI & mei_key).fetch1('mei', download_path=fetch_download_path)
        inner_mei=torch.load(inner_mei_path)
        
        center_mask= (MEIThresholdMask & mei_key & dict(mask_hash=inner_mask_hash)).fetch1( "mask", download_path=fetch_download_path)
        self.center_mask=torch.tensor(center_mask)

        self.centerimg = inner_mei[0][:2]*self.center_mask
        self.originalMEI = inner_mei

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape.
        maintain the behavior channel same as source center MEI"""
        initial = self._create_random_tensor(*shape)
        behav_channel = self.originalMEI
        # only change color channel
        behav_channel[:,self.selected_channels,...]=self.centerimg + initial[:,self.selected_channels,...] * (1-self.center_mask)
        return behav_channel

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"

class BlurAndCutSurr:
    """ Blur an image with a Gaussian window.

    Arguments:
        sigma (float or tuple): Standard deviation in y, x used for the gaussian blurring.
        decay_factor (float): Compute sigma every iteration as `sigma + decay_factor *
            (iteration - 1)`. Ignored if None.
        truncate (float): Gaussian window is truncated after this number of standard
            deviations to each side. Size of kernel = 8 * sigma + 1
        pad_mode (string): Mode for the padding used for the blurring. Valid values are:
            'constant', 'reflect' and 'replicate'
    """

    def __init__(self, sigma, key, decay_factor=None, truncate=4, pad_mode="reflect", cut_channel=None):
        self.sigma = sigma if isinstance(sigma, tuple) else (sigma,) * 2
        self.decay_factor = decay_factor
        self.truncate = truncate
        self.pad_mode = pad_mode
        self.cut_channel = cut_channel
        if cut_channel is not None:
            print(f"cutting channel: {cut_channel}")

        src_method_fn = key["src_method_fn"]
        inner_ensemble_hash = key["inner_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]

        unit_id = key["unit_id"]
        inner_mask_hash = key['mask_hash']
        #inner_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=inner_ensemble_hash) & dict(method_hash=inner_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        
        #inner_mei=torch.load(inner_mei_path)
        mask_key = dj.AndList([dict(method_fn=src_method_fn),
                     dict(ensemble_hash=inner_ensemble_hash),
                     dict(method_hash=inner_method_hash),
                     dict(unit_id=unit_id),dict(mask_hash=inner_mask_hash)])

        center_mask= (MEIThresholdMask & mask_key).fetch1( "mask", download_path=fetch_download_path)
        self.center_mask=torch.tensor(center_mask)

    @varargin
    def __call__(self, x, iteration=None):
        num_channels = x.shape[1]

        # Update sigma if needed
        if self.decay_factor is None:
            sigma = self.sigma
        else:
            sigma = tuple(s + self.decay_factor * (iteration - 1) for s in self.sigma)

        # Define 1-d kernels to use for blurring
        y_halfsize = max(int(round(sigma[0] * self.truncate)), 1)
        y_gaussian = signal.gaussian(2 * y_halfsize + 1, std=sigma[0])
        x_halfsize = max(int(round(sigma[1] * self.truncate)), 1)
        x_gaussian = signal.gaussian(2 * x_halfsize + 1, std=sigma[1])
        y_gaussian = torch.as_tensor(y_gaussian, device=x.device, dtype=x.dtype)
        x_gaussian = torch.as_tensor(x_gaussian, device=x.device, dtype=x.dtype)

        # Blur
        padded_x = F.pad(x, pad=(x_halfsize, x_halfsize, y_halfsize, y_halfsize), mode=self.pad_mode)
        blurred_x = F.conv2d(padded_x, y_gaussian.repeat(num_channels, 1, 1)[..., None], groups=num_channels)
        blurred_x = F.conv2d(blurred_x, x_gaussian.repeat(num_channels, 1, 1, 1), groups=num_channels)
        final_x = blurred_x / (y_gaussian.sum() * x_gaussian.sum())  # normalize
        if self.cut_channel is not None:
            final_x[:, self.cut_channel, ...] *= 0
        final_x[:,:2,...]=final_x[:,:2,...] * (1-self.center_mask).to(x.device)

        return final_x

class ClipNormInChannelSurr:
    """ Change the norm of the input for surround part only.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, channel, key, norm=None, x_min=None, x_max=None):
        self.channel = channel
        self.norm = norm
        self.x_min = x_min
        self.x_max = x_max

        src_method_fn = key["src_method_fn"]
        inner_ensemble_hash = key["inner_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        unit_id = key["unit_id"]
        inner_mask_hash = key['mask_hash']

        mask_key = dj.AndList([dict(method_fn=src_method_fn),
                     dict(ensemble_hash=inner_ensemble_hash),
                     dict(method_hash=inner_method_hash),
                     dict(unit_id=unit_id),dict(mask_hash=inner_mask_hash)])

        center_mask= (MEIThresholdMask & mask_key).fetch1( "mask", download_path=fetch_download_path)        
        self.center_mask=torch.tensor(center_mask)

    @varargin
    def __call__(self, x, iteration=None):
        x_norm = torch.norm(x[:, self.channel, ...] * (1-self.center_mask).to(x.device) )
        if self.norm is not None:
            if x_norm > self.norm:
                x[:, self.channel, ...] = x[:, self.channel, ...] * (1-self.center_mask).to(x.device) * (self.norm / x_norm) + x[:, self.channel, ...] * self.center_mask.to(x.device)
        if self.x_min is not None:
            x[:, self.channel, ...] = torch.clamp(x[:, self.channel, ...], self.x_min, self.x_max)
        return x


class ClipNormInChannelSurr_totalctr:
    """ Change the norm of the input for the total MEI norm.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, channel, key, norm=None, x_min=None, x_max=None):
        self.channel = channel
        self.norm = norm
        self.x_min = x_min
        self.x_max = x_max

        src_method_fn = key["src_method_fn"]
        inner_ensemble_hash = key["inner_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        unit_id = key["unit_id"]
        inner_mask_hash = key['mask_hash']

        mei_key = dj.AndList([dict(method_fn=src_method_fn),
                     dict(ensemble_hash=inner_ensemble_hash),
                     dict(method_hash=inner_method_hash),
                     dict(unit_id=unit_id)])

        inner_mei_path = (MEI & mei_key).fetch1('mei', download_path=fetch_download_path)
        inner_mei=torch.load(inner_mei_path)
        
        center_mask= (MEIThresholdMask & mei_key & dict(mask_hash=inner_mask_hash)).fetch1( "mask", download_path=fetch_download_path)
        self.center_mask=torch.tensor(center_mask)

        self.centerimg = inner_mei[0][:2]*self.center_mask

    @varargin
    def __call__(self, x, iteration=None):
        x_norm = torch.norm(x[:, self.channel, ...] * (1-self.center_mask).to(x.device) )
        x[:, self.channel, ...] = x[:, self.channel, ...] * (1-self.center_mask).to(x.device) * (self.norm / x_norm) + x[:, self.channel, ...] * self.center_mask.to(x.device)
        
        # norm for (center + blurred surround):
        c_norm = torch.norm(x[:, self.channel, ...] * (self.center_mask > 0).to(x.device))
        # norm for nonblurred surround:
        s_norm = torch.norm(x[:, self.channel, ...] * (self.center_mask == 0).to(x.device))
        print(c_norm,' ,,, ',s_norm)
        if self.norm is not None:
            ######!!!!!!!
            norm_surr=torch.sqrt(self.norm**2 - c_norm**2)
            # only rescale nonblurred surround region
            x[:, self.channel, ...] = x[:, self.channel, ...] * (self.center_mask == 0).to(x.device) * (norm_surr / s_norm) + x[:, self.channel, ...] * (self.center_mask>0).to(x.device)
        if self.x_min is not None:
            x[:, self.channel, ...] = torch.clamp(x[:, self.channel, ...], self.x_min, self.x_max)
        return x

class ClipNormInChannelSurr_ctr:
    """ Change the norm of the surround such that total norm is stable.
        works better than "ClipNormInChannelSurr"

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, channel, key, norm=None, x_min=None, x_max=None):
        self.channel = channel
        self.norm = norm
        self.x_min = x_min
        self.x_max = x_max

        src_method_fn = key["src_method_fn"]
        inner_ensemble_hash = key["inner_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        unit_id = key["unit_id"]
        inner_mask_hash = key['mask_hash']


        mei_key = dj.AndList([dict(method_fn=src_method_fn),
                     dict(ensemble_hash=inner_ensemble_hash),
                     dict(method_hash=inner_method_hash),
                     dict(unit_id=unit_id)])

        inner_mei_path = (MEI & mei_key).fetch1('mei', download_path=fetch_download_path)
        inner_mei=torch.load(inner_mei_path)
        
        center_mask= (MEIThresholdMask & mei_key & dict(mask_hash=inner_mask_hash)).fetch1( "mask", download_path=fetch_download_path)
        self.center_mask=torch.tensor(center_mask)

        self.centerimg = inner_mei[0][:2]*self.center_mask

    @varargin
    def __call__(self, x, iteration=None):
        x_norm = torch.norm(x[:, self.channel, ...] * (1-self.center_mask).to(x.device) )        
        x[:, self.channel, ...] = x[:, self.channel, ...] * (1-self.center_mask).to(x.device) * (self.norm / x_norm) + self.centerimg.to(x.device)
        if self.x_min is not None:
            x[:, self.channel, ...] = torch.clamp(x[:, self.channel, ...], self.x_min, self.x_max)
        return x

################################ TRANSFORMS ##############################################
class RenormSurround:
    """ Change the norm of the surround such that total norm is stable.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, channel, key, norm=None):
        self.channel = channel
        self.norm = norm

        src_method_fn = key["src_method_fn"]
        inner_ensemble_hash = key["inner_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        unit_id = key["unit_id"]
        inner_mask_hash = key['mask_hash']


        mei_key = dj.AndList([dict(method_fn=src_method_fn),
                     dict(ensemble_hash=inner_ensemble_hash),
                     dict(method_hash=inner_method_hash),
                     dict(unit_id=unit_id)])

        inner_mei_path = (MEI & mei_key).fetch1('mei', download_path=fetch_download_path)
        inner_mei=torch.load(inner_mei_path)
        
        center_mask= (MEIThresholdMask & mei_key & dict(mask_hash=inner_mask_hash)).fetch1( "mask", download_path=fetch_download_path)
        self.center_mask=torch.tensor(center_mask)

        self.centerimg = inner_mei[0][:2]*self.center_mask

    @varargin
    def __call__(self, x, iteration=None):
        x_norm = torch.norm(x[:, self.channel, ...] * (1-self.center_mask).to(x.device) )  
        
        #print('MEI gradien',x.grad)
        renorm_x = torch.clone(x).requires_grad_(True)
        renorm_x[:, self.channel, ...] = x[:, self.channel, ...] * (1-self.center_mask).to(x.device) * (self.norm / x_norm) + self.centerimg.to(x.device)
        #print('MEI gradient',x.grad)
        #print('rescaled MEI gradient',renorm_x)
        ### here x should be related to renorm_x.grad???
        # y=(2*x+2)
        #print("y is leaf:,", y.is_leaf)
        print("transformed x is leaf: ",renorm_x.is_leaf)
        return renorm_x

################################ POSTPROCESSING ##############################################
class ClipSurround:
    """ Change the norm of the surround such that total norm is stable.
        works better than "ClipNormInChannelSurr"

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """
    def __init__(self, channel, x_min=None, x_max=None):
        self.channel = channel
        self.x_min = x_min
        self.x_max = x_max

    @varargin
    def __call__(self, x, iteration=None):
        x[:, self.channel, ...] = torch.clamp(x[:, self.channel, ...], self.x_min, self.x_max)
        return x


