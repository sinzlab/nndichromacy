import warnings
import numpy as np

import torch
import torch.nn.functional as F
from scipy import signal
from collections.abc import Iterable

from mei.legacy.utils import varargin
from ..tables.scores import MEINorm, MEINormBlue, MEINormGreen
from nndichromacy.tables.from_mei import MEI
import os
fetch_download_path = os.environ.get('FETCH_DOWNLOAD_PATH', '/data/fetched_from_attach')

class BlurAndCut:
    """Blur an image with a Gaussian window.

    Arguments:
        sigma (float or tuple): Standard deviation in y, x used for the gaussian blurring.
        decay_factor (float): Compute sigma every iteration as `sigma + decay_factor *
            (iteration - 1)`. Ignored if None.
        truncate (float): Gaussian window is truncated after this number of standard
            deviations to each side. Size of kernel = 8 * sigma + 1
        pad_mode (string): Mode for the padding used for the blurring. Valid values are:
            'constant', 'reflect' and 'replicate'
    """

    def __init__(
        self, sigma, decay_factor=None, truncate=4, pad_mode="reflect", cut_channel=None
    ):
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
        padded_x = F.pad(
            x, pad=(x_halfsize, x_halfsize, y_halfsize, y_halfsize), mode=self.pad_mode
        )
        blurred_x = F.conv2d(
            padded_x,
            y_gaussian.repeat(num_channels, 1, 1)[..., None],
            groups=num_channels,
        )
        blurred_x = F.conv2d(
            blurred_x, x_gaussian.repeat(num_channels, 1, 1, 1), groups=num_channels
        )
        final_x = blurred_x / (y_gaussian.sum() * x_gaussian.sum())  # normalize
        if self.cut_channel is not None:
            final_x[:, self.cut_channel, ...] *= 0
        return final_x

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

    def __init__(self, sigma, key, decay_factor=None, truncate=4, pad_mode="reflect", cut_channel=None,mask_thres=0.3):
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

        inner_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=inner_ensemble_hash) & dict(method_hash=inner_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        
        inner_mei=torch.load(inner_mei_path)
        self.center_mask= (inner_mei[0][-1] > mask_thres) * 1

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

class ChangeNormAndClip:
    """Change the norm of the input.

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
    """Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, key, method_key, channel, x_min, x_max):
        norm_key = dict()
        norm_key.update(key)
        norm_key["method_hash"] = method_key
        self.norm = (
            (MEINormGreen & norm_key).fetch1("mei_norm")
            if channel == 0
            else (MEINormBlue & norm_key).fetch1("mei_norm")
        )
        self.x_min = x_min
        self.x_max = x_max

    @varargin
    def __call__(self, x, iteration=None):
        x_norm = torch.norm(x.view(len(x), -1), dim=-1)
        renorm = x * (self.norm / x_norm).view(len(x), *[1] * (x.dim() - 1))
        return torch.clamp(renorm, self.x_min, self.x_max)


class ChangeNormAdaptiveAndClip:
    """Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, key, method_key, channel, x_min, x_max):
        norm_key = dict()
        norm_key.update(key)
        norm_key["method_hash"] = method_key
        self.channel = channel
        self.norm = (
            (MEINormGreen & norm_key).fetch1("mei_norm")
            if self.channel == 0
            else (MEINormBlue & norm_key).fetch1("mei_norm")
        )
        self.x_min = x_min
        self.x_max = x_max

    @varargin
    def __call__(self, x, iteration=None):
        x_norm = torch.norm(x[:, self.channel, ...])
        x[:, self.channel, ...] = x[:, self.channel, ...] * (self.norm / x_norm)
        return torch.clamp(x, self.x_min, self.x_max)


class ChangeNormInChannel:
    """Change the norm of the input.

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
    """Change the norm of the input.

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
            x[:, self.channel, ...] = torch.clamp(
                x[:, self.channel, ...], self.x_min, self.x_max
            )
            return x


class ClipNormInChannelAdaptivePupil:
    """Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(
        self,
        channel,
        norm,
        key,
        threshold_percentile,
        pupil_channel,
        x_min=None,
        x_max=None,
    ):
        self.channel = channel
        self.norm = norm
        self.x_min = x_min
        self.x_max = x_max
        self.pupil_channel = pupil_channel

        from ..tables.from_nnfabrik import Dataset

        dataloaders = (Dataset & key).get_dataloader()
        behaviors = []
        for b in dataloaders["train"][key["data_key"]]:
            behaviors.append(b.behavior.cpu().numpy())
        behaviors = np.vstack(behaviors)
        self.min_pupil = np.percentile(behaviors[:, 0], 0)
        self.max_pupil = np.percentile(behaviors[:, 0], threshold_percentile)

    @varargin
    def __call__(self, x, iteration=None):

        x[:, self.pupil_channel, ...] = np.random.uniform(
            self.min_pupil, self.max_pupil
        )
        x_norm = torch.norm(x[:, self.channel, ...])
        if x_norm > self.norm:
            x[:, self.channel, ...] = x[:, self.channel, ...] * (self.norm / x_norm)
        if self.x_min is None:
            return x
        else:
            x[:, self.channel, ...] = torch.clamp(
                x[:, self.channel, ...], self.x_min, self.x_max
            )
            return x

class ClipNormInChannelSurr:
    """ Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, channel, key, norm=None, x_min=None, x_max=None, mask_thres=0.3):
        self.channel = channel
        self.norm = norm
        self.x_min = x_min
        self.x_max = x_max

        src_method_fn = key["src_method_fn"]
        inner_ensemble_hash = key["inner_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        unit_id = key["unit_id"]

        inner_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=inner_ensemble_hash) & dict(method_hash=inner_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        inner_mei=torch.load(inner_mei_path)
        self.center_mask=(inner_mei[0][-1] > mask_thres) * 1

    @varargin
    def __call__(self, x, iteration=None):
        x_norm = torch.norm(x[:, self.channel, ...] * (1-self.center_mask).to(x.device) )
        if self.norm is not None:
            if x_norm > self.norm:
                x[:, self.channel, ...] = x[:, self.channel, ...] * (1-self.center_mask).to(x.device) * (self.norm / x_norm) + x[:, self.channel, ...] * self.center_mask.to(x.device)
        if self.x_min is not None:
            x[:, self.channel, ...] = torch.clamp(x[:, self.channel, ...], self.x_min, self.x_max)
        return x

class ClipNormInEveryChannel:
    """ Change the norm of the input for every specified channel when generate transparent color MEI.

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
        for ch in range(len(self.channel)):
            ch_norm = torch.norm(x[:, self.channel[ch], ...])
            if self.norm[ch] != None:
                if ch_norm > self.norm[ch]: # when actual norm larger than desired norm
                    x[:, self.channel[ch], ...] = x[:, self.channel[ch], ...] * (self.norm[ch] / ch_norm)
            if self.x_min[ch] != None:
                x[:, self.channel[ch], ...] = torch.clamp(x[:, self.channel[ch], ...], self.x_min[ch], self.x_max[ch])
        return x

class ClipNormInChannelforRingFlexNorm:
    """ Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, channel, fullnorm, key, mask_thres_for_ring=0.3, x_min=None, x_max=None):
        self.channel = channel
        self.fullnorm = fullnorm
        self.x_min = x_min
        self.x_max = x_max

        src_method_fn = key["src_method_fn"]
        inner_ensemble_hash = key["inner_ensemble_hash"]
        outer_ensemble_hash = key["outer_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        outer_method_hash = key["outer_method_hash"]
        unit_id = key["unit_id"]

        outer_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=outer_ensemble_hash) & dict(method_hash=outer_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        inner_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=inner_ensemble_hash) & dict(method_hash=inner_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        
        outer_mei=torch.load(outer_mei_path)
        inner_mei=torch.load(inner_mei_path)

        self.ring_mask=(outer_mei[0][1] - inner_mei[0][1] > mask_thres_for_ring) * 1

    @varargin
    def __call__(self, x, iteration=None):
        inner = x[:, self.channel, ...] * (1 - self.ring_mask.to(x.device))
        inner_norm = torch.norm(inner)

        ring = x[:, self.channel, ...] * self.ring_mask.to(x.device)
        ring_norm = torch.norm(ring) 
        
        new_ring = ring * ( torch.sqrt(self.fullnorm**2-inner_norm**2) /ring_norm )
        x[:, self.channel, ...] = new_ring + inner
        #print(self.ring_mask)
        print(torch.norm(new_ring).item(),' + ',torch.norm(inner) ,' == ',torch.norm(new_ring+inner))
        if self.x_min is None:
            return x
        else:
           x[:, self.channel, ...] = torch.clamp(x[:, self.channel, ...], self.x_min, self.x_max)
        return x

class ClipNormInChannelforRing:
    """ Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, channel, norm, key, mask_thres_for_ring=0.3, x_min=None, x_max=None):
        self.channel = channel
        self.norm = norm
        self.x_min = x_min
        self.x_max = x_max

        src_method_fn = key["src_method_fn"]
        inner_ensemble_hash = key["inner_ensemble_hash"]
        outer_ensemble_hash = key["outer_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        outer_method_hash = key["outer_method_hash"]
        unit_id = key["unit_id"]

        outer_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=outer_ensemble_hash) & dict(method_hash=outer_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        inner_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=inner_ensemble_hash) & dict(method_hash=inner_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        
        outer_mei=torch.load(outer_mei_path)
        inner_mei=torch.load(inner_mei_path)

        self.ring_mask=(outer_mei[0][1] - inner_mei[0][1] > mask_thres_for_ring) * 1

    @varargin
    def __call__(self, x, iteration=None):

        x_norm = torch.norm(x[:, self.channel, ...] * self.ring_mask.to(x.device)) 
        if x_norm > self.norm:
            x[:, self.channel, ...] = x[:, self.channel, ...] * self.ring_mask.to(x.device) * (self.norm / x_norm)  + x[:, self.channel, ...] * (1 - self.ring_mask.to(x.device))
        if self.x_min is None:
            return x
        else:
            x[:, self.channel, ...] = torch.clamp(x[:, self.channel, ...], self.x_min, self.x_max)
            return x

class ClipNormInChannelforSurround:
    """ Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, channel1, key, mask_thres=0.3,norm_ch1=None, channel2=None,norm_ch2=None, x_min_ch1=None, x_max_ch1=None, x_min_ch2=None, x_max_ch2=None):

        self.channel1 = channel1
        self.norm_ch1 = norm_ch1
        self.x_min_ch1 = x_min_ch1
        self.x_max_ch1 = x_max_ch1
        
        self.channel2 = channel2
        if self.channel2 is not None:
            self.norm_ch2 = norm_ch2
            self.x_min_ch2 = x_min_ch2
            self.x_max_ch2 = x_max_ch2

        src_method_fn = key["src_method_fn"]
        inner_ensemble_hash = key["inner_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        unit_id = key["unit_id"]

        inner_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=inner_ensemble_hash) & dict(method_hash=inner_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        inner_mei=torch.load(inner_mei_path)
        self.center_mask=(inner_mei[0][1] > mask_thres) * 1

    @varargin
    def __call__(self, x, iteration=None):
        x_norm_ch1 = torch.norm(x[:, self.channel1, ...] * (1-self.center_mask).to(x.device)) 
        if self.norm_ch1 is not None:
            if x_norm_ch1 > self.norm_ch1:
                x[:, self.channel1, ...] = x[:, self.channel1, ...] * (1-self.center_mask).to(x.device) * (self.norm_ch1 / x_norm_ch1)  + x[:, self.channel1, ...] * self.center_mask.to(x.device)
        if self.x_min_ch1 or self.x_max_ch1 is not None:
            x[:, self.channel1, ...] = torch.clamp(x[:, self.channel1, ...], self.x_min_ch1, self.x_max_ch1)

        # when there is transparent channel
        if self.channel2 is not None:
            x[:, self.channel2, ...] = torch.clamp(x[:, self.channel2, ...], self.x_min_ch2, self.x_max_ch2)
        return x

class ClipNormInChannelforCenter:
    """ Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, channel1, key, mask_thres=0.3,norm_ch1=None, channel2=None,norm_ch2=None, x_min_ch1=None, x_max_ch1=None, x_min_ch2=None, x_max_ch2=None):

        self.channel1 = channel1
        self.norm_ch1 = norm_ch1
        self.x_min_ch1 = x_min_ch1
        self.x_max_ch1 = x_max_ch1
        
        self.channel2 = channel2
        if self.channel2 is not None:
            self.norm_ch2 = norm_ch2
            self.x_min_ch2 = x_min_ch2
            self.x_max_ch2 = x_max_ch2

        src_method_fn = key["src_method_fn"]
        inner_ensemble_hash = key["inner_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        unit_id = key["unit_id"]

        inner_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=inner_ensemble_hash) & dict(method_hash=inner_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        
        inner_mei=torch.load(inner_mei_path)

        self.center_mask=(inner_mei[0][1] > mask_thres) * 1

    @varargin
    def __call__(self, x, iteration=None):

        x_norm_ch1 = torch.norm(x[:, self.channel1, ...] * (self.center_mask).to(x.device)) 
        if self.norm_ch1 is not None:
            if x_norm_ch1 > self.norm_ch1:
                x[:, self.channel1, ...] = x[:, self.channel1, ...] * (self.center_mask).to(x.device) * (self.norm_ch1 / x_norm_ch1)
        if self.x_min_ch1 or self.x_max_ch1 is not None:
            x[:, self.channel1, ...] = torch.clamp(x[:, self.channel1, ...], self.x_min_ch1, self.x_max_ch1)

        # when there is transparent channel
        if self.channel2 is not None:
            x[:, self.channel2, ...] = torch.clamp(x[:, self.channel2, ...], self.x_min_ch2, self.x_max_ch2)
        return x

class ClipNormInAllChannel: 
    """ When need add transparency to visualization, change the norm of the input for different channel separately (i.e. color channel & transparent channel)

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, channel1, norm_ch1,channel2=None,norm_ch2=None, x_min_ch1=None, x_max_ch1=None,
                         x_min_ch2=None, x_max_ch2=None):
        self.channel1 = channel1
        self.norm_ch1 = norm_ch1
        self.x_min_ch1 = x_min_ch1
        self.x_max_ch1 = x_max_ch1
        
        self.channel2 = channel2
        if self.channel2 is not None:
            self.norm_ch2 = norm_ch2
            self.x_min_ch2 = x_min_ch2
            self.x_max_ch2 = x_max_ch2

    @varargin
    def __call__(self, x, iteration=None):
        x_norm_ch1 = torch.norm(x[:, self.channel1, ...])
        x_norm_ch2 = torch.norm(x[:, self.channel2, ...])

        if x_norm_ch1 > self.norm_ch1:
            x[:, self.channel1, ...] = x[:, self.channel1, ...] * (self.norm_ch1 / x_norm_ch1)

        if self.x_min_ch1 or self.x_max_ch1 is not None:
            x[:, self.channel1, ...] = torch.clamp(x[:, self.channel1, ...], self.x_min_ch1, self.x_max_ch1)

        # when there is transparent channel
        if self.channel2 is not None:
            if self.norm_ch2 is not None:
                if x_norm_ch2 > self.norm_ch2:
                    x[:, self.channel2, ...] = x[:, self.channel2, ...] * (self.norm_ch2 / x_norm_ch2)

            x[:, self.channel2, ...] = torch.clamp(x[:, self.channel2, ...], self.x_min_ch2, self.x_max_ch2)
        return x

class ChangeNormShuffleBehavior:
    """Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(
        self,
        channel,
        norm,
        first_behav_channel,
        pupil_limits,
        dpupil_limits,
        treadmill_limits,
    ):

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
        x[:, self.first_behav_channel, ...] = np.random.uniform(
            self.pupil_limits[0], self.pupil_limits[1]
        )
        x[:, self.first_behav_channel + 1, ...] = np.random.uniform(
            self.dpupil_limits[0], self.dpupil_limits[1]
        )
        x[:, self.first_behav_channel + 2, ...] = np.random.uniform(
            self.treadmill_limits[0], self.treadmill_limits[1]
        )
        return x

class ChangeStdClampedMean:
    """Change the norm of the input.

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
        x = x.clamp(self.x_min, self.x_max)
        x_std = torch.std(x.view(len(x), -1), dim=-1)

        # set x to have the desired std
        x = x * (self.std / (x_std + 1e-9)).view(len(x), *[1] * (x.dim() - 1))
        # compute mean of x
        x_mean = torch.mean(x.view(len(x), -1), dim=-1)
        # set mean to the clamped value
        x = x + (self.clamped_mean - x_mean).view(len(x), *[1] * (x.dim() - 1))
        return x
