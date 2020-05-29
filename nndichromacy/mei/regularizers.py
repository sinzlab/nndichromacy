from featurevis.legacy.ops import ChangeStd, GaussianBlur, Jitter, TotalVariation, ChangeNorm, ClipRange
from featurevis.legacy.utils import Compose
from featurevis import utils, ops
from .utility import cumstom_initial_guess
from functools import partial


# Contrast
postup_contrast_01 = ChangeStd(0.1)
postup_contrast_1 = ChangeStd(1)
postup_contrast_125 = ChangeStd(12.5)
postup_contrast_100 = ChangeStd(10)
postup_contrast_5 = ChangeStd(5)

# Blurring
Blur_sigma_1 = GaussianBlur(1)


# Regularizers of DiCarlo 2019
jitter_DiCarlo = Jitter(max_jitter=(2, 4))
total_variation_DiCarlo = TotalVariation(weight=0.001)
gradient_DiCarlo = Compose([ChangeNorm(1), ClipRange(-1, 1)])

# Initial Guess
rgb_initial_guess = partial(cumstom_initial_guess, mean=111, std=60)

# walker regularizers
walker_gradient = utils.Compose([ops.FourierSmoothing(0.04), # not exactly the same as fft_smooth(precond=0.1) but close
                                 ops.DivideByMeanOfAbsolute(),
                                 ops.MultiplyBy(1/850, decay_factor=(1/850 - 1/20400) /(1-1000))])  # decays from 1/850 to 1/20400 in 1000 iterations
bias, scale = 111.28329467773438, 60.922306060791016
walker_postup = utils.Compose([ops.ClipRange(-bias / scale, (255 - bias) / scale),
                               ops.GaussianBlur(1.5, decay_factor=(1.5 - 0.01) /(1-1000))]) # decays from 1.5 to 0.01 in 1000 iterations