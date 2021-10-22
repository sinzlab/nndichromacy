from . import ops
from . import utils

from .ops import ChangeStd, GaussianBlur

postup_contrast_01 = ChangeStd(0.1)
postup_contrast_1 = ChangeStd(1)
Blur_sigma_1 = GaussianBlur(1)
