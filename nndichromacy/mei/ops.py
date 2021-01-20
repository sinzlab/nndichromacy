import warnings
import numpy as np

import torch
import torch.nn.functional as F
from scipy import signal
from collections.abc import Iterable

from mei.legacy.utils import varargin
from ..tables.scores import MEINorm, MEINormBlue, MEINormGreen


class BlurAndCut:
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

    def __init__(self, sigma, decay_factor=None, truncate=4, pad_mode="reflect", cut_channel=None):
        self.sigma = sigma if isinstance(sigma, tuple) else (sigma,) * 2
        self.decay_factor = decay_factor
        self.truncate = truncate
        self.pad_mode = pad_mode
        self.cut_channel = cut_channel
        if cut_channel is not None:
            print(f"cutting channel: {cut_channel}")

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

        return final_x


class ChangeNormAndClip:
    """ Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, norm, x_min, x_max):
        self.norm = norm
        self.x_min = x_min
        self.x_max = x_max

    @varargin
    def __call__(self, x, iteration=None):
        x_norm = torch.norm(x.view(len(x), -1), dim=-1)
        renorm = x * (self.norm / x_norm).view(len(x), *[1] * (x.dim() - 1))
        return torch.clamp(renorm, self.x_min, self.x_max)


class ChangeNormAndClipAdaptive:
    """ Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, key, method_key, channel, x_min, x_max):
        norm_key = dict()
        norm_key.update(key)
        norm_key["method_hash"] = method_key
        self.norm = (MEINormGreen & norm_key).fetch1("mei_norm") if channel == 0 else (MEINormBlue & norm_key).fetch1("mei_norm")
        self.x_min = x_min
        self.x_max = x_max

    @varargin
    def __call__(self, x, iteration=None):
        x_norm = torch.norm(x.view(len(x), -1), dim=-1)
        renorm = x * (self.norm / x_norm).view(len(x), *[1] * (x.dim() - 1))
        return torch.clamp(renorm, self.x_min, self.x_max)


class ChangeNormAdaptiveAndClip:
    """ Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, key, method_key, channel, x_min, x_max):
        norm_key = dict()
        norm_key.update(key)
        norm_key["method_hash"] = method_key
        self.channel = channel
        self.norm = (MEINormGreen & norm_key).fetch1("mei_norm") if self.channel == 0 else (MEINormBlue & norm_key).fetch1("mei_norm")
        self.x_min = x_min
        self.x_max = x_max

    @varargin
    def __call__(self, x, iteration=None):
        x_norm = torch.norm(x[:, self.channel, ...])
        x[:, self.channel, ...] = x[:, self.channel, ...] * (self.norm / x_norm)
        return torch.clamp(x, self.x_min, self.x_max)


class ChangeNormInChannel:
    """ Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, channel, norm):

        self.channel = channel
        self.norm = norm

    @varargin
    def __call__(self, x, iteration=None):
        x_norm = torch.norm(x[:, self.channel, ...])
        x[:, self.channel, ...] = x[:, self.channel, ...] * (self.norm / x_norm)
        return x


class ClipNormInChannel:
    """ Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, channel, norm, x_min=None, x_max=None):
        self.channel = channel
        self.norm = norm
        self.x_min = x_min
        self.x_max = x_max

    @varargin
    def __call__(self, x, iteration=None):
        x_norm = torch.norm(x[:, self.channel, ...])
        if x_norm > self.norm:
            x[:, self.channel, ...] = x[:, self.channel, ...] * (self.norm / x_norm)
        if self.x_min is None:
            return x
        else:
            return torch.clamp(x, self.x_min, self.x_max)

class ChangeNormShuffleBehavior:
    """ Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self,
                 channel,
                 norm,
                 first_behav_channel,
                 pupil_limits,
                 dpupil_limits,
                 treadmill_limits):

        self.channel = channel
        self.norm = norm
        self.first_behav_channel = first_behav_channel
        self.pupil_limits = pupil_limits
        self.dpupil_limits = dpupil_limits
        self.treadmill_limits = treadmill_limits

    @varargin
    def __call__(self, x, iteration=None):
        x_norm = torch.norm(x[:, self.channel, ...])
        x[:, self.channel, ...] = x[:, self.channel, ...] * (self.norm / x_norm)
        x[:, self.first_behav_channel, ...] = np.random.uniform(self.pupil_limits[0], self.pupil_limits[1])
        x[:, self.first_behav_channel+1, ...] = np.random.uniform(self.dpupil_limits[0], self.dpupil_limits[1])
        x[:, self.first_behav_channel+2, ...] = np.random.uniform(self.treadmill_limits[0], self.treadmill_limits[1])
        return x


class ChangeStdClampedMean:
    """ Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, std, x_min, x_max, clamped_mean):
        self.clamped_mean = clamped_mean
        self.std = std
        self.x_min = x_min
        self.x_max = x_max
        self.clamped_mean = clamped_mean

    @varargin
    def __call__(self, x, iteration=None):
        x = x.clamp(self.x_min,  self.x_max)
        x_std = torch.std(x.view(len(x), -1), dim=-1)

        # set x to have the desired std
        x = x * (self.std / (x_std + 1e-9)).view(len(x), *[1] * (x.dim() - 1))
        # compute mean of x
        x_mean = torch.mean(x.view(len(x), -1), dim=-1)
        # set mean to the clamped value
        x = x + (self.clamped_mean - x_mean).view(len(x), *[1] * (x.dim() - 1))
        return x
