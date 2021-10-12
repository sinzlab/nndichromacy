from nndichromacy.legacy.featurevis.ops import (
    ChangeStd,
    GaussianBlur,
    Jitter,
    TotalVariation,
    ChangeNorm,
    ClipRange,
)
from nndichromacy.legacy.featurevis.utils import Compose
from nndichromacy.legacy.featurevis import utils, ops
from .utility import cumstom_initial_guess
from functools import partial


# Contrast
postup_contrast_01 = ChangeStd(0.1)
postup_contrast_1 = ChangeStd(1)
postup_contrast_125 = ChangeStd(12.5)
postup_contrast_100 = ChangeStd(10)
postup_contrast_5 = ChangeStd(5)

postup_contrast_0075 = ChangeStd(0.075)
postup_contrast_0125 = ChangeStd(0.125)
postup_contrast_015 = ChangeStd(0.15)
postup_contrast_0175 = ChangeStd(0.175)
postup_contrast_02 = ChangeStd(0.2)
postup_contrast_0225 = ChangeStd(0.225)
postup_contrast_025 = ChangeStd(0.25)
postup_contrast_0275 = ChangeStd(0.275)
postup_contrast_03 = ChangeStd(0.3)
postup_contrast_0325 = ChangeStd(0.325)
postup_contrast_035 = ChangeStd(0.35)


# Blurring
Blur_sigma_1 = GaussianBlur(1)


# Regularizers of DiCarlo 2019
jitter_DiCarlo = Jitter(max_jitter=(2, 4))
total_variation_DiCarlo = TotalVariation(weight=0.001)
gradient_DiCarlo = Compose([ChangeNorm(1), ClipRange(-1, 1)])

# Initial Guess
rgb_initial_guess = partial(cumstom_initial_guess, mean=111, std=60)

# walker regularizers
walker_gradient = utils.Compose(
    [
        ops.FourierSmoothing(
            0.04
        ),  # not exactly the same as fft_smooth(precond=0.1) but close
        ops.DivideByMeanOfAbsolute(),
        ops.MultiplyBy(1 / 850, decay_factor=(1 / 850 - 1 / 20400) / (1 - 1000)),
    ]
)  # decays from 1/850 to 1/20400 in 1000 iterations
bias, scale = 111.28329467773438, 60.922306060791016
walker_postup = utils.Compose(
    [
        ops.ClipRange(-bias / scale, (255 - bias) / scale),
        ops.GaussianBlur(1.5, decay_factor=(1.5 - 0.01) / (1 - 1000)),
    ]
)  # decays from 1.5 to 0.01 in 1000 iterations
