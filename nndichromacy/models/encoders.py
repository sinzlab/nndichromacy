import numpy as np
import torch
import copy

from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, core, readout, final_nonlinearity, elu_offset, shifter=None):
        super().__init__()
        self.core = core
        self.readout = readout
        self.offset = elu_offset
        self.shifter = shifter
        self.readout_nonlinearity = final_nonlinearity

    def forward(
        self, *args, data_key=None, eye_pos=None, shift=None, trial_idx=None, **kwargs
    ):

        x = self.core(args[0])
        if len(args) > 2:
            for j in range(2, len(args)):
                if hasattr(args[j], "shape"):
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

            # import ipdb; ipdb.set_trace()
            if trial_idx is not None:
                if not isinstance(trial_idx, torch.Tensor):
                    trial_idx = torch.tensor(trial_idx)
                trial_idx = trial_idx.to(x.device).to(dtype=x.dtype)

                # ipdb.set_trace()
                if self.shifter[data_key].mlp[0].in_features == 3:
                    eye_pos = torch.cat((eye_pos, trial_idx), dim=1)

            shift = self.shifter[data_key](eye_pos)

        x = self.readout(x, data_key=data_key, shift=shift, **kwargs)
        if self.readout_nonlinearity is True:
            return F.elu(x + self.offset) + 1
        else:
            return x

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

        sample = kwargs["sample"] if "sample" in kwargs else None
        x = self.readout(x, data_key=data_key, sample=sample)
        return F.elu(x + self.offset) + 1

    def regularizer(self, data_key):
        return self.core.regularizer() + self.readout.regularizer(data_key=data_key)


class GeneralEncoder(nn.Module):
    def __init__(self, core, readout, elu_offset, shifter=None):
        super().__init__()
        self.core = core
        self.readout = readout
        self.offset = elu_offset
        self.shifter = shifter

    def forward(self, *args, data_key=None, **kwargs):
        x = self.core(args[0])

        trial_idx = kwargs.get("trial_idx", None)
        pupil_center = kwargs.get("pupil_center", None)
        eye_pos = kwargs.get("eye_pos", None)

        behavior = kwargs.pop("behavior", None)
        if behavior is not None:
            if not isinstance(behavior, torch.Tensor):
                behavior = torch.tensor(behavior)
            behavior = behavior.to(x.device).to(dtype=x.dtype)

        if eye_pos is not None and self.shifter is not None:
            if not isinstance(eye_pos, torch.Tensor):
                eye_pos = torch.tensor(eye_pos)

            # overwrite pupil_center
            pupil_center = eye_pos.to(x.device).to(dtype=x.dtype)

        shift = kwargs.get("shift", None)

        if pupil_center is not None and self.shifter is not None:
            if not isinstance(pupil_center, torch.Tensor):
                pupil_center = torch.tensor(pupil_center)
            pupil_center = pupil_center.to(x.device).to(dtype=x.dtype)
            if trial_idx is not None:
                if not isinstance(trial_idx, torch.Tensor):
                    trial_idx = torch.tensor(trial_idx)
                trial_idx = trial_idx.to(x.device).to(dtype=x.dtype)

                if self.shifter[data_key].mlp[0].in_features == 3:
                    pupil_center = torch.cat((pupil_center, trial_idx), dim=1)

            shift = self.shifter[data_key](pupil_center)

        x = self.readout(x, data_key=data_key, shift=shift, behavior=behavior, **kwargs)
        return F.elu(x + self.offset) + 1

    def regularizer(self, data_key):
        return self.core.regularizer() + self.readout.regularizer(data_key=data_key)
