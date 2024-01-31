import warnings

import torch
import torch.nn.functional as F
from scipy import signal
from neuralpredictors.regularizers import LaplaceL2

from mei.legacy.utils import varargin
from torch import Tensor
from nnfabrik import builder
import random
import numpy as np

from nndichromacy.tables.from_mei import MEI
import os
fetch_download_path = os.environ.get('FETCH_DOWNLOAD_PATH', '/data/fetched_from_attach')
################################## REGULARIZERS ##########################################
class TotalVariation:
    """ Total variation regularization.

    Arguments:
        weight (float): Weight of the regularization.
        isotropic (bool): Whether to use the isotropic or anisotropic definition of Total
            Variation. Default is anisotropic (l1-norm of the gradient).
    """

    def __init__(self, weight=1, isotropic=False):
        self.weight = weight
        self.isotropic = isotropic

    @varargin
    def __call__(self, x, iteration=None):
        # Using the definitions from Wikipedia.
        diffs_y = torch.abs(x[:, :, 1:] - x[:, :, -1:])
        diffs_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        if self.isotropic:
            tv = (
                torch.sqrt(diffs_y[:, :, :, :-1] ** 2 + diffs_x[:, :, :-1, :] ** 2).reshape(len(x), -1).sum(-1)
            )  # per image
        else:
            tv = diffs_y.reshape(len(x), -1).sum(-1) + diffs_x.reshape(len(x), -1).sum(-1)  # per image
        loss = self.weight * torch.mean(tv)

        return loss

class RegTransparency:
    """ transparency regularization, by adjusting weight to control the transparent level.

    Arguments:
        weight (float): Weight of the regularization.

    """

    def __init__(self, weight=0):
        self.weight = weight

    @varargin
    def __call__(self, x, iteration=None):
        mean_alpha_value = x
        loss = self.weight * mean_alpha_value

        return loss

class LpNorm:
    """Computes the lp-norm of an input.

    Arguments:
        weight (float): Weight of the regularization
        p (int): Degree for the l-p norm.
    """

    def __init__(self, weight=1, p=6):
        self.weight = weight
        self.p = p

    @varargin
    def __call__(self, x, iteration=None):
        lpnorm = (torch.abs(x) ** self.p).reshape(len(x), -1).sum(-1) ** (1 / self.p)
        loss = self.weight * torch.mean(lpnorm)
        return loss


class Similarity:
    """ Compute similarity metrics across all examples in one batch.

    Arguments:
        weight (float): Weight of the regularization.
        metric (str): What metric to use when computing pairwise similarities. One of:
            correlation: Masked correlation.
            cosine: Cosine similarity of the masked input.
            neg_euclidean: Negative of euclidean distance between the masked input.
        combine_op (function): Function used to agglomerate pairwise similarities.
        mask (torch.tensor or None): Mask to use when calculating similarities. Expected
            to be in [0, 1] range and be broadcastable with input.
    """

    def __init__(self, weight=1, metric="correlation", combine_op=torch.max, mask=None):
        self.weight = weight
        self.metric = metric
        self.combine_op = combine_op
        self.mask = mask

    @varargin
    def __call__(self, x, iteration=None):
        if len(x) < 2:
            warnings.warn("Only one image in the batch. Similarity regularization will" "return 0")
            return 0

        # Mask x
        masked_x = x if self.mask is None else x * self.mask
        flat_x = masked_x.view(len(x), -1)

        # Compute similarity matrix
        if self.metric == "correlation":
            if self.mask is None:
                residuals = flat_x - flat_x.mean(-1, keepdim=True)
                numer = torch.mm(residuals, residuals.t())
                ssr = (residuals ** 2).sum(-1)
            else:
                mask_sum = self.mask.sum() * (flat_x.shape[-1] / len(self.mask.view(-1)))
                mean = flat_x.sum(-1) / mask_sum
                residuals = x - mean.view(len(x), *[1] * (x.dim() - 1))  # N x 1 x 1 x 1
                numer = (residuals[None, :] * residuals[:, None] * self.mask).view(len(x), len(x), -1).sum(-1)
                ssr = ((residuals ** 2) * self.mask).view(len(x), -1).sum(-1)
            sim_matrix = numer / (torch.sqrt(torch.ger(ssr, ssr)) + 1e-9)
        elif self.metric == "cosine":
            norms = torch.norm(flat_x, dim=-1)
            sim_matrix = torch.mm(flat_x, flat_x.t()) / (torch.ger(norms, norms) + 1e-9)
        elif self.metric == "neg_euclidean":
            sim_matrix = -torch.norm(flat_x[None, :, :] - flat_x[:, None, :], dim=-1)
        else:
            raise ValueError("Invalid metric name:{}".format(self.metric))

        # Compute overall similarity
        triu_idx = torch.triu(torch.ones(len(x), len(x)), diagonal=1) == 1
        similarity = self.combine_op(sim_matrix[triu_idx])

        loss = self.weight * similarity

        return loss


class BoxContrast():
    def __init__(self, weight=0.1, filter_size=7, box_constraint=10, p=2, padding=0, l1_weight=1e-3):
        self.weight = weight
        self.filter_size = filter_size
        self.box_constraint = box_constraint
        self.p = p
        self.box_filter = torch.ones([1, 1, filter_size, filter_size])
        self.ReLU = torch.nn.ReLU()
        self.padding = padding
        self.l1_regularizer = LpNorm(weight=l1_weight, p=1)

    @varargin
    def __call__(self, x, iteration=None):
        box_loss = self.weight * torch.mean(self.ReLU(F.conv2d(x**2, self.box_filter.to(x.device), padding=self.padding)
                                                 - self.box_constraint) ** self.p)
        l1_loss = self.l1_regularizer(x)
        return box_loss + l1_loss

class BoxContrastPixelL2():
    def __init__(self, weight=0.1, upper=2.0, lower=-1.5, p=2, l2_weight=1, l1_weight=0, filter_size=3):
        self.weight = weight
        self.upper = upper
        self.lower = lower
        self.p = p
        self.ReLU = torch.nn.ReLU()
        self.l2_regularizer = LaplaceL2(filter_size=filter_size)
        self.l2_weight = l2_weight
        self.l1_regularizer = LpNorm(weight=l1_weight, p=1)

    @varargin
    def __call__(self, x, iteration=None):
        pixel_loss = self.weight * torch.sum((self.ReLU(x - self.upper) ** self.p)
                                              + self.ReLU(-(x - self.lower) ** self.p))

        l2_loss = self.l2_weight * self.l2_regularizer.to(x.device)(x, avg=True)
        l1_loss = self.l1_regularizer(x)
        return pixel_loss + l2_loss + l1_loss


# class PixelCNN():
#     def __init__(self, weight=1):
#         self.weight = weight
#
#         self.pixel_cnn = ... # load the model
#
#     @varargin
#     def __call__(self, x):
#         # Modify x to make it a valid input to pixel cnn (add channels)
#         prior = self.pixel_cnn(x)
#         loss = self.weight * prior

class Transparency():
    # to encourage transparency
    # weight means the encouraging factor
    def __init__(self,weight=0.5):
        self.weight = weight

    @varargin
    def __call__(self, x, iteration=None):
        opacity = torch.mean( x[:,-1,...])
        return self.weight*opacity

class NatImgBackground():
    def __init__(self,dataset_fn,dataset_path,norm=None,dataset_name='22564-3-12'):
        self.dataset_fn = dataset_fn
        self.dataset_path = dataset_path
        self.dataset_config = {'paths': dataset_path,
                 'normalize': True,
                 'include_behavior': False,
                 'batch_size': 128,
                 'exclude': None,
                 'file_tree': True,
                  'scale':1
                 }
        self.dataset_name = dataset_name
        self.images=None
        self.norm=norm

    @varargin
    def __call__(self, x,iteration=None):
        if iteration==0:
            dataloaders = builder.get_data(self.dataset_fn, self.dataset_config)
            images = []
            for tier in ['train','test','validation']:
                for i,j in dataloaders[tier][self.dataset_name]:
                    images.append(i.squeeze().data)
                #responses.append(j.squeeze().cpu().data.numpy())
            self.images = torch.vstack(images)

        bg=random.choice(self.images)
        if self.norm is not None:
            normed_bg = bg * (self.norm / torch.norm(bg))
            bg = torch.clamp(normed_bg,-1.96, 2.12) 
        return bg

class NatImgBackgroundHighNorm():
    # to get a more robust mask, use high norm background; 
    # but to help MEI converge, increasing background norm as the iteration step increase
    def __init__(self,dataset_fn,dataset_path,start_norm=None,end_norm=None,dataset_name='22564-3-12'):
        #self.dataset_fn = dataset_fn
        #self.dataset_path = dataset_path
        dataset_config = {'paths': dataset_path,
                 'normalize': True,
                 'include_behavior': False,
                 'batch_size': 128,
                 'exclude': None,
                 'file_tree': True,
                  'scale':1
                 }
        dataset_name = dataset_name
        dataloaders = builder.get_data(dataset_fn, dataset_config)
        images = []
        for tier in ['train','test','validation']:
            for i,j in dataloaders[tier][dataset_name]:
                images.append(i.squeeze().data)
        self.images = torch.vstack(images)
        self.start_norm=start_norm
        self.end_norm=end_norm

    @varargin
    def __call__(self, x,iteration=None):

        bg=random.choice(self.images)
        normed_bg = bg * (self.start_norm+iteration*(self.end_norm-self.start_norm)/1000.0 / torch.norm(bg))
        #bg = torch.clamp(normed_bg,-1.96, 2.12) 
        return normed_bg
    
class WhiteNoiseBackground():
    def __init__(self, mean, std,shape=(72,128),strength=1):
        self.mean = mean
        self.std = std
        self.shape = shape
        self.strength = strength

    @varargin
    def __call__(self, x, iteration=None):
        bg_img=np.random.normal(self.mean, self.std, self.shape).astype('f')
        #print(bg_img)
        #bg_img = np.clip(bg_img, -1.96, 2.12)
        rang=max(max(bg_img.flatten()),abs(min(bg_img.flatten())))
        bg_img=bg_img/rang*2.12*self.strength # such that each pixel range in (-2.12,2.12)
        return torch.as_tensor(bg_img)

################################ TRANSFORMS ##############################################
class Jitter:
    """ Jitter the image at random by some certain amount.

    Arguments:
        max_jitter(tuple of ints): Maximum amount of jitter in y, x.
    """

    def __init__(self, max_jitter):
        self.max_jitter = max_jitter if isinstance(max_jitter, tuple) else (max_jitter, max_jitter)

    @varargin
    def __call__(self, x, iteration=None):
        # Sample how much to jitter
        jitter_y = torch.randint(-self.max_jitter[0], self.max_jitter[0] + 1, (1,), dtype=torch.int32).item()
        jitter_x = torch.randint(-self.max_jitter[1], self.max_jitter[1] + 1, (1,), dtype=torch.int32).item()

        # Pad and crop the rest
        pad_y = (jitter_y, 0) if jitter_y >= 0 else (0, -jitter_y)
        pad_x = (jitter_x, 0) if jitter_x >= 0 else (0, -jitter_x)
        padded_x = F.pad(x, pad=(*pad_x, *pad_y), mode="reflect")

        # Crop
        h, w = x.shape[-2:]
        jittered_x = padded_x[
            ...,
            slice(0, h) if jitter_y > 0 else slice(-jitter_y, None),
            slice(0, w) if jitter_x > 0 else slice(-jitter_x, None),
        ]

        return jittered_x


class RandomCrop:
    """ Take a random crop of the input image.

    Arguments:
        height (int): Height of the crop.
        width (int): Width of the crop
    """

    def __init__(self, height, width):
        self.height = height
        self.width = width

    @varargin
    def __call__(self, x, iteration=None):
        crop_y = torch.randint(0, max(0, x.shape[-2] - self.height) + 1, (1,), dtype=torch.int32).item()
        crop_x = torch.randint(0, max(0, x.shape[-1] - self.width) + 1, (1,), dtype=torch.int32).item()
        cropped_x = x[..., crop_y : crop_y + self.height, crop_x : crop_x + self.width]

        return cropped_x


class BatchedCrops:
    """ Create a batch of crops of the original image.

    Arguments:
        height (int): Height of the crop
        width (int): Width of the crop
        step_size (int or tuple): Number of pixels in y, x to step for each crop.
        sigma (float or tuple): Sigma in y, x for the gaussian mask applied to each batch.
            None to avoid masking

    Note:
        Increasing the stride of every convolution to stride * step_size produces the same
        effect in a much more memory efficient way but it will be architecture dependent
        and may not play nice with the rest of transforms.
    """

    def __init__(self, height, width, step_size, sigma=None):
        self.height = height
        self.width = width
        self.step_size = step_size if isinstance(step_size, tuple) else (step_size,) * 2
        self.sigma = sigma if sigma is None or isinstance(sigma, tuple) else (sigma,) * 2

        # If needed, create gaussian mask
        if sigma is not None:
            y_gaussian = signal.gaussian(height, std=self.sigma[0])
            x_gaussian = signal.gaussian(width, std=self.sigma[1])
            self.mask = y_gaussian[:, None] * x_gaussian

    @varargin
    def __call__(self, x, iteration=None):
        if len(x) > 1:
            raise ValueError("x can only have one example.")
        if x.shape[-2] < self.height or x.shape[-1] < self.width:
            raise ValueError("x should be larger than the expected crop")

        # Take crops
        crops = []
        for i in range(0, x.shape[-2] - self.height + 1, self.step_size[0]):
            for j in range(0, x.shape[-1] - self.width + 1, self.step_size[1]):
                crops.append(x[..., i : i + self.height, j : j + self.width])
        crops = torch.cat(crops, dim=0)

        # Multiply by a gaussian mask if needed
        if self.sigma is not None:
            mask = torch.as_tensor(self.mask, device=crops.device, dtype=crops.dtype)
            crops = crops * mask

        return crops


class ChangeRange:
    """ This changes the range of x as follows:
        new_x = sigmoid(x) * (desired_max - desired_min) + desired_min

    Arguments:
        x_min (float or tensor): Minimum desired value for the output. If a tensor it
            needs to be broadcastable with x.
        x_max (float or tensor): Minimum desired value for the output. If a tensor it
            needs to be broadcastable with x.
    """

    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    @varargin
    def __call__(self, x, iteration=None):
        new_x = torch.sigmoid(x) * (self.x_max - self.x_min) + self.x_min
        return new_x


class Resize:
    """ Resize images.

    Arguments:
        scale_factor (float): Factor to rescale the images:
            new_h, new_w = round(scale_factor * (old_h, old_w)).
        resize_method (str): 'nearest' or 'bilinear' interpolation.

    Note:
        This changes the dimensions of the image.
    """

    def __init__(self, scale_factor, resize_method="bilinear"):
        self.scale_factor = scale_factor
        self.resample_method = resize_method

    @varargin
    def __call__(self, x, iteration=None):
        new_height = int(round(x.shape[-2] * self.scale_factor))
        new_width = int(round(x.shape[-1] * self.scale_factor))
        return F.upsample(x, (new_height, new_width), mode=self.resize_method)


class GrayscaleToRGB:
    """ Transforms a single channel image into three channels (by copying the channel)."""

    @varargin
    def __call__(self, x,iteration=None):
        if x.dim() != 4 or x.shape[1] != 1:
            raise ValueError("Image is not grayscale!")

        return x.expand(-1, 3, -1, -1)


class Identity:
    """ Transform that returns the input as is."""

    @varargin
    def __call__(self, x, iteration=None):
        return x



############################## GRADIENT OPERATIONS #######################################
class ChangeNorm:
    """ Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, norm):
        self.norm = norm

    @varargin
    def __call__(self, x, iteration=None):
        x_norm = torch.norm(x.view(len(x), -1), dim=-1)
        renorm = x * (self.norm / x_norm).view(len(x), *[1] * (x.dim() - 1))
        return renorm


class ClipRange:
    """Clip the value of x to some specified range.

    Arguments:
        x_min (float): Lower valid value.
        x_max (float): Higher valid value.
    """

    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    @varargin
    def __call__(self, x, iteration=None):
        return torch.clamp(x, self.x_min, self.x_max)


class FourierSmoothing:
    """ Smooth the input in the frequency domain.

    Image is transformed to fourier domain, power densities at i, j are multiplied by
    (1 - ||f||)**freq_exp where ||f|| = sqrt(f_i**2 + f_j**2) and the image is brought
    back to the spatial domain:
        new_x = ifft((1 - freqs) ** freq_exp * fft(x))

    Arguments:
        freq_exp (float): Exponent for the frequency mask. Higher numbers produce more
            smoothing.

    Note:
        Consider just using Gaussian blurring. Faster and easier to explain.
    """

    def __init__(self, freq_exp):
        self.freq_exp = freq_exp

    @varargin
    def __call__(self, x, iteration=None):
        # Create mask of frequencies (following np.fft.rfftfreq and np.fft.fftfreq docs)
        h, w = x.shape[-2:]
        freq_y = (
            torch.cat(
                [torch.arange((h - 1) // 2 + 1, dtype=torch.float32), -torch.arange(h // 2, 0, -1, dtype=torch.float32)]
            )
            / h
        )  # fftfreq
        freq_x = torch.arange(w // 2 + 1, dtype=torch.float32) / w  # rfftfreq
        yx_freq = torch.sqrt(freq_y[:, None] ** 2 + freq_x ** 2)

        # Create smoothing mask
        norm_freq = yx_freq * torch.sqrt(torch.tensor(2.0))  # 0-1
        mask = (1 - norm_freq) ** self.freq_exp

        # Smooth
        freq = torch.rfft(x, signal_ndim=2)  # same output as np.fft.rfft2
        mask = torch.as_tensor(mask, device=freq.device, dtype=freq.dtype).unsqueeze(-1)
        smooth = torch.irfft(freq * mask, signal_ndim=2, signal_sizes=x.shape[-2:])
        return smooth


class DivideByMeanOfAbsolute:
    """ Divides x by the mean of absolute x. """

    @varargin
    def __call__(self, x, iteration=None):
        return x / torch.abs(x).view(len(x), -1).mean(-1)


class MultiplyBy:
    """Multiply x by some constant.

    Arguments:
        const: Number x will be multiplied by
        decay_factor: Compute const every iteration as `const + decay_factor * (iteration
            - 1)`. Ignored if None.
    """

    def __init__(self, const, decay_factor=None):
        self.const = const
        self.decay_factor = decay_factor

    @varargin
    def __call__(self, x, iteration=None):
        if self.decay_factor is None:
            const = self.const
        else:
            const = self.const + self.decay_factor * (iteration - 1)

        return const * x


########################### POST UPDATE OPERATIONS #######################################
class GaussianBlurforRing:
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

    def __init__(self, sigma, key, decay_factor=None, truncate=4, mask_thres_for_ring=0.3, pad_mode="reflect"):
        
        self.sigma = sigma if isinstance(sigma, tuple) else (sigma,) * 2
        self.decay_factor = decay_factor
        self.truncate = truncate
        self.pad_mode = pad_mode
 
        # To get ring mask from key and MEI table
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
        num_channels = x.shape[1]
        padded_x = F.pad(x, pad=(x_halfsize, x_halfsize, y_halfsize, y_halfsize), mode=self.pad_mode)
        blurred_x = F.conv2d(padded_x, y_gaussian.repeat(num_channels, 1, 1)[..., None], groups=num_channels)
        blurred_x = F.conv2d(blurred_x, x_gaussian.repeat(num_channels, 1, 1, 1), groups=num_channels)
        final_x = blurred_x / (y_gaussian.sum() * x_gaussian.sum())  # normalize

        return final_x * self.ring_mask.to(x.device)

class GaussianBlurforCenter:
    """ Blur an image with a Gaussian window for surround region"""
    """ only blur for the center """

    def __init__(self, sigma, key, decay_factor=None, truncate=4, mask_thres=0.3, pad_mode="reflect"):
        
        self.sigma = sigma if isinstance(sigma, tuple) else (sigma,) * 2
        self.decay_factor = decay_factor
        self.truncate = truncate
        self.pad_mode = pad_mode
 
        # To get center mask from key and MEI table
        src_method_fn = key["src_method_fn"]
        inner_ensemble_hash = key["inner_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]

        unit_id = key["unit_id"]

        inner_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=inner_ensemble_hash) & dict(method_hash=inner_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        
        inner_mei=torch.load(inner_mei_path)
        self.center_mask= (inner_mei[0][1] > mask_thres) * 1

    @varargin
    def __call__(self, x, iteration=None):
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
        num_channels = x.shape[1]
        padded_x = F.pad(x, pad=(x_halfsize, x_halfsize, y_halfsize, y_halfsize), mode=self.pad_mode)
        blurred_x = F.conv2d(padded_x, y_gaussian.repeat(num_channels, 1, 1)[..., None], groups=num_channels)
        blurred_x = F.conv2d(blurred_x, x_gaussian.repeat(num_channels, 1, 1, 1), groups=num_channels)
        final_x = blurred_x / (y_gaussian.sum() * x_gaussian.sum())  # normalize

        return final_x * (self.center_mask).to(x.device)

class GaussianBlurforSurround:
    """ Blur an image with a Gaussian window for surround region
    """

    def __init__(self, sigma, key, decay_factor=None, truncate=4, mask_thres=0.3, pad_mode="reflect"):
        
        self.sigma = sigma if isinstance(sigma, tuple) else (sigma,) * 2
        self.decay_factor = decay_factor
        self.truncate = truncate
        self.pad_mode = pad_mode
 
        # To get ring mask from key and MEI table
        src_method_fn = key["src_method_fn"]
        inner_ensemble_hash = key["inner_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]

        unit_id = key["unit_id"]

        inner_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=inner_ensemble_hash) & dict(method_hash=inner_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        
        inner_mei=torch.load(inner_mei_path)
        self.center_mask= (inner_mei[0][1] > mask_thres) * 1

    @varargin
    def __call__(self, x, iteration=None):
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
        num_channels = x.shape[1]
        padded_x = F.pad(x, pad=(x_halfsize, x_halfsize, y_halfsize, y_halfsize), mode=self.pad_mode)
        blurred_x = F.conv2d(padded_x, y_gaussian.repeat(num_channels, 1, 1)[..., None], groups=num_channels)
        blurred_x = F.conv2d(blurred_x, x_gaussian.repeat(num_channels, 1, 1, 1), groups=num_channels)
        final_x = blurred_x / (y_gaussian.sum() * x_gaussian.sum())  # normalize

        return final_x * (1-self.center_mask).to(x.device)

class GaussianBlur:
    """ Blur an image with a Gaussian window.

    Arguments:
        sigma (float or tuple): Standard deviation in y, x used for the gaussian blurring.
        decay_factor (float): Compute sigma every iteration as `sigma + decay_factor *
            (iteration - 1)`. Ignored if None.
        truncate (float): Gaussian window is truncated after this number of standard
            deviations to each side. Size of kernel = 8 * sigma + 1
        pad_mode (string): Mode for the padding used for the blurring. Valid values are:
            'constant', 'reflect' and 'replicate'
        mei_only (True/False): for transparent mei, if True, no Gaussian blur for transparent channel:
            default should be False (also for non transparent case)
    """

    def __init__(self, sigma, decay_factor=None, truncate=4, pad_mode="reflect",mei_only=False):
        self.sigma = sigma if isinstance(sigma, tuple) else (sigma,) * 2
        self.decay_factor = decay_factor
        self.truncate = truncate
        self.pad_mode = pad_mode
        self.mei_only = mei_only

    @varargin
    def __call__(self, x, iteration=None):
        

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
        if self.mei_only:
            num_channels = x.shape[1]-1
            padded_x = F.pad(x[:,:-1,...], pad=(x_halfsize, x_halfsize, y_halfsize, y_halfsize), mode=self.pad_mode)
        else: # also blur transparent channel
            num_channels = x.shape[1]
            padded_x = F.pad(x, pad=(x_halfsize, x_halfsize, y_halfsize, y_halfsize), mode=self.pad_mode)
        blurred_x = F.conv2d(padded_x, y_gaussian.repeat(num_channels, 1, 1)[..., None], groups=num_channels)
        blurred_x = F.conv2d(blurred_x, x_gaussian.repeat(num_channels, 1, 1, 1), groups=num_channels)
        final_x = blurred_x / (y_gaussian.sum() * x_gaussian.sum())  # normalize
        # print(final_x.shape)
        if self.mei_only:
            return torch.cat((final_x,x[:,-1,...].view(x.shape[0],1,x.shape[2],x.shape[3])),dim=1)
        else:
            return final_x


class ChangeStd:
    """ Change the standard deviation of input.

        Arguments:
        std (float or tensor): Desired std. If tensor, it should be the same length as x.
        zero_mean (boolean):   If False, the mean of x will be preserved after the std is changed. Defaults to False,
                                   such that the mean will is preserved by default.
    """

    def __init__(self, std, zero_mean=True):
        self.std = std
        self.zero_mean = zero_mean

    @varargin
    def __call__(self, x, iteration=None):
        x_std = torch.std(x.view(len(x), -1), dim=-1)
        x_mean = torch.mean(x.view(len(x), -1), dim=-1)
        fixed_std = x * (self.std / (x_std + 1e-9)).view(len(x), *[1] * (x.dim() - 1))

        x_mean_rescaled = torch.mean(fixed_std.view(len(fixed_std), -1), dim=-1)
        rescaled_x = fixed_std + (x_mean - x_mean_rescaled).view(len(x), *[1] * (x.dim() - 1))

        return fixed_std if self.zero_mean else rescaled_x

