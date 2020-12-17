import numpy as np
import torch
import copy

from neuralpredictors.layers.cores import Stacked2dCore, RotationEquivariant2dCore
from neuralpredictors.layers.legacy import Gaussian2d
from neuralpredictors.layers.readouts import PointPooled2d
from nnfabrik.utility.nn_helpers import get_module_output, set_random_seed, get_dims_for_loader_dict
from torch import nn
from torch.nn import functional as F

from .cores import SE2dCore, TransferLearningCore
from .readouts import MultipleFullGaussian2d, MultiReadout, MultipleSpatialXFeatureLinear
from .utility import unpack_data_info
from .utility import *
from .shifters import MLPShifter, StaticAffine2dShifter


class Encoder(nn.Module):

    def __init__(self, core, readout, elu_offset, shifter=None):
        super().__init__()
        self.core = core
        self.readout = readout
        self.offset = elu_offset
        self.shifter = shifter


    def forward(self, *args, data_key=None, eye_pos=None, shift=None, trial_idx=None, **kwargs):

        x = self.core(args[0])
        if len(args) > 2:
            for j in range(2, len(args)):
                if hasattr(args[j], 'shape'):
                    if len(args[j].shape) == 2:
                        if args[j].shape[1] == 2:
                            eye_pos = args[j]
                        elif args[j].shape[1] == 1:
                            trial_idx = args[j]

        if "pupil_center" in kwargs:
            eye_pos = kwargs["pupil_center"]

        if "trial_idx" in kwargs:
            trial_idx = kwargs["trial_idx"]

        if eye_pos is not None and self.shifter is not None:
            if not isinstance(eye_pos, torch.Tensor):
                eye_pos = torch.tensor(eye_pos)
            eye_pos = eye_pos.to(x.device).to(dtype=x.dtype)

            #import ipdb; ipdb.set_trace()
            if trial_idx is not None:
                if not isinstance(trial_idx, torch.Tensor):
                    trial_idx = torch.tensor(trial_idx)
                trial_idx = trial_idx.to(x.device).to(dtype=x.dtype)

                #ipdb.set_trace()
                if self.shifter[data_key].mlp[0].in_features == 3:
                    eye_pos = torch.cat((eye_pos, trial_idx), dim=1)

            shift = self.shifter[data_key](eye_pos)

        if "sample" in kwargs:
            x = self.readout(x, data_key=data_key, sample=kwargs["sample"], shift=shift)
        else:
            x = self.readout(x, data_key=data_key, shift=shift)

        return F.elu(x + self.offset) + 1

    def regularizer(self, data_key):
        return self.core.regularizer() + self.readout.regularizer(data_key=data_key)


class EncoderShifter(nn.Module):

    def __init__(self, core, readout, shifter, elu_offset):
        super().__init__()
        self.core = core
        self.readout = readout
        self.offset = elu_offset
        self.shifter = shifter


    def forward(self, *args, data_key=None, eye_position=None, **kwargs):

        x = self.core(args[0])

        if len(args) == 3:
            if args[2].shape[1] == 2:
                eye_position = args[2]

        if len(args) == 4:
            if args[3].shape[1] == 2:
                eye_position = args[3]


        sample = kwargs["sample"] if 'sample' in kwargs else None
        x = self.readout(x, data_key=data_key, sample=sample)
        return F.elu(x + self.offset) + 1

    def regularizer(self, data_key):
        return self.core.regularizer() + self.readout.regularizer(data_key=data_key)