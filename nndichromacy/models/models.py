import numpy as np
import torch
import copy

from neuralpredictors.layers.cores import Stacked2dCore, RotationEquivariant2dCore
from neuralpredictors.layers.legacy import Gaussian2d
from neuralpredictors.layers.readouts import PointPooled2d, FullGaussian2d
from nnfabrik.utility.nn_helpers import set_random_seed, get_dims_for_loader_dict
from neuralpredictors.utils import get_module_output
from torch import nn
from torch.nn import functional as F

from .utility import *
from .shifters import MLPShifter, StaticAffine2dShifter
from .encoders import Encoder, GeneralEncoder
from .cores import SE2dCore, TransferLearningCore
from .readouts import (
    MultipleFullGaussian2d,
    MultiReadout,
    MultipleSpatialXFeatureLinear,
    MultipleGaussian2d,
    MultipleFullGaussian2dBehav,
)

try:
    from ..tables.from_nnfabrik import TrainedModel
    from ..tables.from_mei import TrainedEnsembleModel
    from nnfabrik.main import Model
except ModuleNotFoundError:
    pass
except:
    print("dj database connection could not be established. no access to pretrained models available.")


# from . import logger as log


class MultiplePointPooled2d(MultiReadout, torch.nn.ModuleDict):
    def __init__(
        self,
        core,
        in_shape_dict,
        n_neurons_dict,
        pool_steps,
        pool_kern,
        bias,
        init_range,
        gamma_readout,
    ):
        # super init to get the _module attribute
        super(MultiplePointPooled2d, self).__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(
                k,
                PointPooled2d(
                    in_shape,
                    n_neurons,
                    pool_steps=pool_steps,
                    pool_kern=pool_kern,
                    bias=bias,
                    init_range=init_range,
                ),
            )
        self.gamma_readout = gamma_readout


class MultipleGaussian2d(torch.nn.ModuleDict):
    def __init__(
        self,
        core,
        in_shape_dict,
        n_neurons_dict,
        init_mu_range,
        init_sigma_range,
        bias,
        gamma_readout,
    ):
        # super init to get the _module attribute
        super(MultipleGaussian2d, self).__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(
                k,
                Gaussian2d(
                    in_shape=in_shape,
                    outdims=n_neurons,
                    init_mu_range=init_mu_range,
                    init_sigma_range=init_sigma_range,
                    bias=bias,
                ),
            )
        self.gamma_readout = gamma_readout

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return self[data_key].feature_l1(average=False) * self.gamma_readout


def se_core_gauss_readout(
    dataloaders,
    seed,
    hidden_channels=32,
    input_kern=13,  # core args
    hidden_kern=3,
    layers=3,
    gamma_input=15.5,
    skip=0,
    final_nonlinearity=True,
    momentum=0.9,
    pad_input=False,
    batch_norm=True,
    hidden_dilation=1,
    laplace_padding=None,
    input_regularizer="LaplaceL2norm",
    init_mu_range=0.2,
    init_sigma_range=0.5,
    readout_bias=True,  # readout args,
    gamma_readout=4,
    elu_offset=0,
    stack=None,
    se_reduction=32,
    n_se_blocks=1,
    depth_separable=False,
    linear=False,
    data_info=None,
):
    """
    Model class of a stacked2dCore (from neuralpredictors) and a pointpooled (spatial transformer) readout

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]

        all other args: See Documentation of Stacked2dCore in neuralpredictors.layers.cores and
            PointPooled2D in neuralpredictors.layers.readouts

    Returns: An initialized model which consists of model.core and model.readout
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = list(input_channels.values())[0] if isinstance(input_channels, dict) else input_channels[0]

    class Encoder(nn.Module):
        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)

            sample = kwargs["sample"] if "sample" in kwargs else None
            x = self.readout(x, data_key=data_key, sample=sample)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

    set_random_seed(seed)

    core = SE2dCore(
        input_channels=core_input_channels,
        hidden_channels=hidden_channels,
        input_kern=input_kern,
        hidden_kern=hidden_kern,
        layers=layers,
        gamma_input=gamma_input,
        skip=skip,
        final_nonlinearity=final_nonlinearity,
        bias=False,
        momentum=momentum,
        pad_input=pad_input,
        batch_norm=batch_norm,
        hidden_dilation=hidden_dilation,
        laplace_padding=laplace_padding,
        input_regularizer=input_regularizer,
        stack=stack,
        se_reduction=se_reduction,
        n_se_blocks=n_se_blocks,
        depth_separable=depth_separable,
        linear=linear,
    )

    readout = MultipleGaussian2d(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        init_mu_range=init_mu_range,
        bias=readout_bias,
        init_sigma_range=init_sigma_range,
        gamma_readout=gamma_readout,
    )

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))[:2]
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def se_core_full_gauss_readout(
    dataloaders,
    seed,
    hidden_channels=32,
    input_kern=13,  # core args
    hidden_kern=3,
    layers=3,
    gamma_input=15.5,
    skip=0,
    final_nonlinearity=True,
    momentum=0.9,
    pad_input=False,
    batch_norm=True,
    hidden_dilation=1,
    laplace_padding=None,
    input_regularizer="LaplaceL2norm",
    init_mu_range=0.2,
    init_sigma=1.0,
    readout_bias=True,  # readout args,
    gamma_readout=4,
    final_readout_nonlinearity=True,
    elu_offset=0,
    stack=None,
    se_reduction=32,
    n_se_blocks=1,
    depth_separable=False,
    linear=False,
    gauss_type="full",
    grid_mean_predictor=None,
    share_features=False,
    share_grid=False,
    data_info=None,
    attention_conv=False,
    shifter=None,
    shifter_type="MLP",
    input_channels_shifter=2,
    hidden_channels_shifter=5,
    shift_layers=3,
    gamma_shifter=0,
    shifter_bias=True,
    hidden_padding=None,
):
    """
    Model class of a stacked2dCore (from neuralpredictors) and a pointpooled (spatial transformer) readout

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]
        isotropic: whether the Gaussian readout should use isotropic Gaussians or not
        grid_mean_predictor: if not None, needs to be a dictionary of the form
            {
            'type': 'cortex',
            'input_dimensions': 2,
            'hidden_layers':0,
            'hidden_features':20,
            'final_tanh': False,
            }
            In that case the datasets need to have the property `neurons.cell_motor_coordinates`
        share_features: whether to share features between readouts. This requires that the datasets
            have the properties `neurons.multi_match_id` which are used for matching. Every dataset
            has to have all these ids and cannot have any more.
        all other args: See Documentation of Stacked2dCore in neuralpredictors.layers.cores and
            PointPooled2D in neuralpredictors.layers.readouts

    Returns: An initialized model which consists of model.core and model.readout
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = list(input_channels.values())[0] if isinstance(input_channels, dict) else input_channels[0]

    source_grids = None
    grid_mean_predictor_type = None
    if grid_mean_predictor is not None:
        grid_mean_predictor = copy.deepcopy(grid_mean_predictor)
        grid_mean_predictor_type = grid_mean_predictor.pop("type")
        if grid_mean_predictor_type == "cortex":
            input_dim = grid_mean_predictor.pop("input_dimensions", 2)
            source_grids = {k: v.dataset.neurons.cell_motor_coordinates[:, :input_dim] for k, v in dataloaders.items()}
        elif grid_mean_predictor_type == "shared":
            pass

    shared_match_ids = None
    if share_features or share_grid:
        shared_match_ids = {k: v.dataset.neurons.multi_match_id for k, v in dataloaders.items()}
        all_multi_unit_ids = set(np.hstack(shared_match_ids.values()))

        for match_id in shared_match_ids.values():
            assert len(set(match_id) & all_multi_unit_ids) == len(
                all_multi_unit_ids
            ), "All multi unit IDs must be present in all datasets"

    if shifter is True:
        data_keys = [i for i in dataloaders.keys()]
        if shifter_type == "MLP":
            shifter = MLPShifter(
                data_keys=data_keys,
                input_channels=input_channels_shifter,
                hidden_channels_shifter=hidden_channels_shifter,
                shift_layers=shift_layers,
                gamma_shifter=gamma_shifter,
            )

        elif shifter_type == "StaticAffine":
            shifter = StaticAffine2dShifter(
                data_keys=data_keys,
                input_channels=input_channels_shifter,
                bias=shifter_bias,
                gamma_shifter=gamma_shifter,
            )

    set_random_seed(seed)

    core = SE2dCore(
        input_channels=core_input_channels,
        hidden_channels=hidden_channels,
        input_kern=input_kern,
        hidden_kern=hidden_kern,
        layers=layers,
        gamma_input=gamma_input,
        skip=skip,
        final_nonlinearity=final_nonlinearity,
        bias=False,
        momentum=momentum,
        pad_input=pad_input,
        batch_norm=batch_norm,
        hidden_dilation=hidden_dilation,
        laplace_padding=laplace_padding,
        input_regularizer=input_regularizer,
        stack=stack,
        se_reduction=se_reduction,
        n_se_blocks=n_se_blocks,
        depth_separable=depth_separable,
        linear=linear,
        attention_conv=attention_conv,
        hidden_padding=hidden_padding,
    )

    readout = MultipleFullGaussian2d(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        init_mu_range=init_mu_range,
        bias=readout_bias,
        init_sigma=init_sigma,
        gamma_readout=gamma_readout,
        gauss_type=gauss_type,
        grid_mean_predictor=grid_mean_predictor,
        grid_mean_predictor_type=grid_mean_predictor_type,
        source_grids=source_grids,
        share_features=share_features,
        share_grid=share_grid,
        shared_match_ids=shared_match_ids,
    )

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))[:2]
            readout[key].bias.data = targets.mean(0)

    model = Encoder(
        core=core,
        readout=readout,
        final_nonlinearity=final_readout_nonlinearity,
        elu_offset=elu_offset,
        shifter=shifter,
    )

    return model


def se_core_behavior_gauss(
    dataloaders,
    seed,
    hidden_channels=32,
    input_kern=13,  # core args
    hidden_kern=3,
    layers=3,
    gamma_input=15.5,
    skip=0,
    final_nonlinearity=True,
    momentum=0.9,
    pad_input=False,
    batch_norm=True,
    hidden_dilation=1,
    laplace_padding=None,
    input_regularizer="LaplaceL2norm",
    init_mu_range=0.2,
    init_sigma=1.0,
    readout_bias=True,  # readout args,
    gamma_readout=4,
    elu_offset=0,
    stack=None,
    se_reduction=32,
    n_se_blocks=1,
    depth_separable=False,
    linear=False,
    gauss_type="full",
    grid_mean_predictor=None,
    share_features=False,
    share_grid=False,
    data_info=None,
    attention_conv=False,
    shifter=None,
    shifter_type="MLP",
    input_channels_shifter=2,
    hidden_channels_shifter=5,
    shift_layers=3,
    gamma_shifter=0,
    shifter_bias=True,
    hidden_padding=None,
    gamma_modulator=None,
    gamma_dissimilarity=None,
):
    """
    Model class of a stacked2dCore (from neuralpredictors) and a pointpooled (spatial transformer) readout

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]
        isotropic: whether the Gaussian readout should use isotropic Gaussians or not
        grid_mean_predictor: if not None, needs to be a dictionary of the form
            {
            'type': 'cortex',
            'input_dimensions': 2,
            'hidden_layers':0,
            'hidden_features':20,
            'final_tanh': False,
            }
            In that case the datasets need to have the property `neurons.cell_motor_coordinates`
        share_features: whether to share features between readouts. This requires that the datasets
            have the properties `neurons.multi_match_id` which are used for matching. Every dataset
            has to have all these ids and cannot have any more.
        all other args: See Documentation of Stacked2dCore in neuralpredictors.layers.cores and
            PointPooled2D in neuralpredictors.layers.readouts

    Returns: An initialized model which consists of model.core and model.readout
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = list(input_channels.values())[0] if isinstance(input_channels, dict) else input_channels[0]

    if "train" in dataloaders.keys():
        dataloaders = dataloaders["train"]
    data_key = list(dataloaders.keys())[0]
    example_batch = next(iter(dataloaders[data_key]))
    behavior = getattr(example_batch, "behavior", None)
    if behavior is None:
        raise ValueError(
            "dataloader does not contain behavioral variables. This model can not be instantiated without behavior"
        )
    behavior_channels = behavior.shape[1]

    source_grids = None
    grid_mean_predictor_type = None
    if grid_mean_predictor is not None:
        grid_mean_predictor = copy.deepcopy(grid_mean_predictor)
        grid_mean_predictor_type = grid_mean_predictor.pop("type")
        if grid_mean_predictor_type == "cortex":
            input_dim = grid_mean_predictor.pop("input_dimensions", 2)
            source_grids = {k: v.dataset.neurons.cell_motor_coordinates[:, :input_dim] for k, v in dataloaders.items()}
        elif grid_mean_predictor_type == "shared":
            pass

    shared_match_ids = None
    if share_features or share_grid:
        shared_match_ids = {k: v.dataset.neurons.multi_match_id for k, v in dataloaders.items()}
        all_multi_unit_ids = set(np.hstack(shared_match_ids.values()))

        for match_id in shared_match_ids.values():
            assert len(set(match_id) & all_multi_unit_ids) == len(
                all_multi_unit_ids
            ), "All multi unit IDs must be present in all datasets"

    if shifter is True:
        data_keys = [i for i in dataloaders.keys()]
        if shifter_type == "MLP":
            shifter = MLPShifter(
                data_keys=data_keys,
                input_channels=input_channels_shifter,
                hidden_channels_shifter=hidden_channels_shifter,
                shift_layers=shift_layers,
                gamma_shifter=gamma_shifter,
            )

        elif shifter_type == "StaticAffine":
            shifter = StaticAffine2dShifter(
                data_keys=data_keys,
                input_channels=input_channels_shifter,
                bias=shifter_bias,
                gamma_shifter=gamma_shifter,
            )

    set_random_seed(seed)

    core = SE2dCore(
        input_channels=core_input_channels,
        hidden_channels=hidden_channels,
        input_kern=input_kern,
        hidden_kern=hidden_kern,
        layers=layers,
        gamma_input=gamma_input,
        skip=skip,
        final_nonlinearity=final_nonlinearity,
        bias=False,
        momentum=momentum,
        pad_input=pad_input,
        batch_norm=batch_norm,
        hidden_dilation=hidden_dilation,
        laplace_padding=laplace_padding,
        input_regularizer=input_regularizer,
        stack=stack,
        se_reduction=se_reduction,
        n_se_blocks=n_se_blocks,
        depth_separable=depth_separable,
        linear=linear,
        attention_conv=attention_conv,
        hidden_padding=hidden_padding,
    )

    readout = MultipleFullGaussian2dBehav(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        init_mu_range=init_mu_range,
        bias=readout_bias,
        init_sigma=init_sigma,
        gamma_readout=gamma_readout,
        gauss_type=gauss_type,
        behavior_channels=behavior_channels,
        grid_mean_predictor=grid_mean_predictor,
        grid_mean_predictor_type=grid_mean_predictor_type,
        source_grids=source_grids,
        share_features=share_features,
        share_grid=share_grid,
        shared_match_ids=shared_match_ids,
        gamma_modulator=gamma_modulator,
        gamma_dissimilarity=gamma_dissimilarity,
    )

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))[:2]
            readout[key].bias.data = targets.mean(0)

    model = GeneralEncoder(core=core, readout=readout, elu_offset=elu_offset, shifter=shifter)

    return model


def se_core_point_readout(
    dataloaders,
    seed,
    hidden_channels=32,
    input_kern=13,  # core args
    hidden_kern=3,
    layers=3,
    gamma_input=15.5,
    skip=0,
    final_nonlinearity=True,
    momentum=0.9,
    pad_input=False,
    batch_norm=True,
    hidden_dilation=1,
    laplace_padding=None,
    input_regularizer="LaplaceL2norm",
    pool_steps=2,
    pool_kern=3,
    init_range=0.2,
    readout_bias=True,  # readout args,
    gamma_readout=4,
    elu_offset=0,
    stack=None,
    se_reduction=32,
    n_se_blocks=1,
    depth_separable=False,
    linear=False,
    data_info=None,
):
    """
    Model class of a stacked2dCore (from neuralpredictors) and a pointpooled (spatial transformer) readout

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]

        all other args: See Documentation of Stacked2dCore in neuralpredictors.layers.cores and
            PointPooled2D in neuralpredictors.layers.readouts

    Returns: An initialized model which consists of model.core and model.readout
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = list(input_channels.values())[0] if isinstance(input_channels, dict) else input_channels[0]

    set_random_seed(seed)

    core = SE2dCore(
        input_channels=core_input_channels,
        hidden_channels=hidden_channels,
        input_kern=input_kern,
        hidden_kern=hidden_kern,
        layers=layers,
        gamma_input=gamma_input,
        skip=skip,
        final_nonlinearity=final_nonlinearity,
        bias=False,
        momentum=momentum,
        pad_input=pad_input,
        batch_norm=batch_norm,
        hidden_dilation=hidden_dilation,
        laplace_padding=laplace_padding,
        input_regularizer=input_regularizer,
        stack=stack,
        se_reduction=se_reduction,
        n_se_blocks=n_se_blocks,
        depth_separable=depth_separable,
        linear=linear,
    )

    readout = MultiplePointPooled2d(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        pool_steps=pool_steps,
        pool_kern=pool_kern,
        bias=readout_bias,
        gamma_readout=gamma_readout,
        init_range=init_range,
    )

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))[:2]
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def stacked2d_core_gaussian_readout(
    dataloaders,
    seed,
    hidden_channels=32,
    input_kern=13,  # core args
    hidden_kern=3,
    layers=3,
    gamma_hidden=0,
    gamma_input=0.1,
    skip=0,
    final_nonlinearity=True,
    core_bias=False,
    momentum=0.9,
    pad_input=False,
    batch_norm=True,
    hidden_dilation=1,
    laplace_padding=None,
    input_regularizer="LaplaceL2norm",
    readout_bias=True,
    init_mu_range=0.2,
    init_sigma_range=0.5,  # readout args,
    gamma_readout=0.1,
    elu_offset=0,
    stack=None,
    isotropic=True,
    data_info=None,
):
    """
    Model class of a stacked2dCore (from neuralpredictors) and a pointpooled (spatial transformer) readout

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]

        all other args: See Documentation of Stacked2dCore in neuralpredictors.layers.cores and
            PointPooled2D in neuralpredictors.layers.readouts

    Returns: An initialized model which consists of model.core and model.readout
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = list(input_channels.values())[0] if isinstance(input_channels, dict) else input_channels[0]

    class Encoder(nn.Module):
        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)
            x = self.readout(x, data_key=data_key, **kwargs)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

    set_random_seed(seed)

    core = Stacked2dCore(
        input_channels=core_input_channels,
        hidden_channels=hidden_channels,
        input_kern=input_kern,
        hidden_kern=hidden_kern,
        layers=layers,
        gamma_hidden=gamma_hidden,
        gamma_input=gamma_input,
        skip=skip,
        final_nonlinearity=final_nonlinearity,
        bias=core_bias,
        momentum=momentum,
        pad_input=pad_input,
        batch_norm=batch_norm,
        hidden_dilation=hidden_dilation,
        laplace_padding=laplace_padding,
        input_regularizer=input_regularizer,
        stack=stack,
    )

    readout = MultipleFullGaussian2d(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        init_mu_range=init_mu_range,
        init_sigma=init_sigma_range,
        bias=readout_bias,
        gamma_readout=gamma_readout,
        gauss_type=isotropic,
    )

    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))[:2]
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def vgg_core_gauss_readout(
    dataloaders,
    seed,
    input_channels=1,
    tr_model_fn="vgg16",  # begin of core args
    model_layer=11,
    momentum=0.1,
    final_batchnorm=True,
    final_nonlinearity=True,
    bias=False,
    init_mu_range=0.4,
    init_sigma_range=0.6,
    readout_bias=True,  # begin or readout args
    gamma_readout=0.002,
    elu_offset=-1,
    gauss_type="uncorrelated",
    data_info=None,
):
    """
    A Model class of a predefined core (using models from torchvision.models). Can be initialized pretrained or random.
    Can also be set to be trainable or not, independent of initialization.

    Args:
        dataloaders: a dictionary of train-dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: ..
        pool_steps:
        pool_kern:
        readout_bias:
        init_range:
        gamma_readout:

    Returns:
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = list(input_channels.values())[0] if isinstance(input_channels, dict) else input_channels[0]

    class Encoder(nn.Module):
        """
        helper nn class that combines the core and readout into the final model
        """

        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)
            x = self.readout(x, data_key=data_key)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.readout.regularizer(data_key=data_key) + self.core.regularizer()

    set_random_seed(seed)

    core = TransferLearningCore(
        input_channels=core_input_channels,
        tr_model_fn=tr_model_fn,
        model_layer=model_layer,
        momentum=momentum,
        final_batchnorm=final_batchnorm,
        final_nonlinearity=final_nonlinearity,
        bias=bias,
    )

    readout = MultipleGaussian2d(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        init_mu_range=init_mu_range,
        bias=readout_bias,
        gamma_readout=gamma_readout,
        init_sigma_range=init_sigma_range,
    )

    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))[:2]
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def vgg_core_full_gauss_readout(
    dataloaders,
    seed,
    input_channels=1,
    tr_model_fn="vgg16",  # begin of core args
    model_layer=11,
    momentum=0.1,
    final_batchnorm=True,
    final_nonlinearity=True,
    bias=False,
    init_mu_range=0.4,
    init_sigma_range=0.6,
    readout_bias=True,  # begin or readout args
    gamma_readout=0.002,
    elu_offset=-1,
    gauss_type="uncorrelated",
    data_info=None,
):
    """
    A Model class of a predefined core (using models from torchvision.models). Can be initialized pretrained or random.
    Can also be set to be trainable or not, independent of initialization.

    Args:
        dataloaders: a dictionary of train-dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: ..
        pool_steps:
        pool_kern:
        readout_bias:
        init_range:
        gamma_readout:

    Returns:
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = list(input_channels.values())[0] if isinstance(input_channels, dict) else input_channels[0]

    class Encoder(nn.Module):
        """
        helper nn class that combines the core and readout into the final model
        """

        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)
            x = self.readout(x, data_key=data_key)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.readout.regularizer(data_key=data_key) + self.core.regularizer()

    set_random_seed(seed)

    core = TransferLearningCore(
        input_channels=core_input_channels,
        tr_model_fn=tr_model_fn,
        model_layer=model_layer,
        momentum=momentum,
        final_batchnorm=final_batchnorm,
        final_nonlinearity=final_nonlinearity,
        bias=bias,
    )

    readout = MultipleGaussian2d(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        init_mu_range=init_mu_range,
        bias=readout_bias,
        init_sigma_range=init_sigma_range,
        gamma_readout=gamma_readout,
    )

    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))[:2]
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def se_core_spatialXfeature_readout(
    dataloaders,
    seed,
    hidden_channels=32,
    input_kern=13,  # core args
    hidden_kern=3,
    layers=3,
    gamma_input=15.5,
    skip=0,
    final_nonlinearity=True,
    momentum=0.9,
    pad_input=False,
    batch_norm=True,
    hidden_dilation=1,
    laplace_padding=None,
    input_regularizer="LaplaceL2norm",
    init_noise=1e-3,
    readout_bias=True,  # readout args,
    gamma_readout=4,
    normalize=False,
    elu_offset=0,
    stack=None,
    se_reduction=32,
    n_se_blocks=1,
    depth_separable=False,
    linear=False,
    data_info=None,
):
    """
    Model class of a stacked2dCore (from neuralpredictors) and a spatialXfeature (factorized) readout

    Args:

    Returns: An initialized model which consists of model.core and model.readout
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = list(input_channels.values())[0] if isinstance(input_channels, dict) else input_channels[0]

    class Encoder(nn.Module):
        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)

            x = self.readout(
                x,
                data_key=data_key,
            )
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

    set_random_seed(seed)

    core = SE2dCore(
        input_channels=core_input_channels,
        hidden_channels=hidden_channels,
        input_kern=input_kern,
        hidden_kern=hidden_kern,
        layers=layers,
        gamma_input=gamma_input,
        skip=skip,
        final_nonlinearity=final_nonlinearity,
        bias=False,
        momentum=momentum,
        pad_input=pad_input,
        batch_norm=batch_norm,
        hidden_dilation=hidden_dilation,
        laplace_padding=laplace_padding,
        input_regularizer=input_regularizer,
        stack=stack,
        se_reduction=se_reduction,
        n_se_blocks=n_se_blocks,
        depth_separable=depth_separable,
        linear=linear,
    )

    readout = MultipleSpatialXFeatureLinear(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        init_noise=init_noise,
        bias=readout_bias,
        gamma_readout=gamma_readout,
        normalize=normalize,
    )

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))[:2]
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def rotation_equivariant_gauss_readout(
    dataloaders,
    seed,
    hidden_channels=32,
    input_kern=13,
    hidden_kern=3,
    layers=3,
    num_rotations=8,
    init_mu_range=0.2,
    init_sigma_range=0.5,
    readout_bias=True,  # readout args,
    gamma_readout=4,
    elu_offset=0,
    data_info=None,
):
    """
    A minimal implementation of the rotation equivariant core.
    """

    if "train" in dataloaders.keys():
        dataloaders = dataloaders["train"]

    # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
    in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

    session_shape_dict = get_dims_for_loader_dict(dataloaders)
    n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
    in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
    input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = list(input_channels.values())[0] if isinstance(input_channels, dict) else input_channels[0]

    class Encoder(nn.Module):
        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)

            sample = kwargs["sample"] if "sample" in kwargs else None
            x = self.readout(x, data_key=data_key, sample=sample)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

    set_random_seed(seed)

    core = RotationEquivariant2dCore(
        input_channels=core_input_channels,
        hidden_channels=hidden_channels,
        input_kern=input_kern,
        hidden_kern=hidden_kern,
        layers=layers,
        num_rotations=num_rotations,
    )

    readout = MultipleGaussian2d(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        init_mu_range=init_mu_range,
        bias=readout_bias,
        init_sigma_range=init_sigma_range,
        gamma_readout=gamma_readout,
    )

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))[:2]
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def augmented_full_readout(
    dataloaders=None,
    seed=None,
    key=None,
    mua_in=False,
    augment_x_start=-0.75,
    augment_x_end=0.75,
    augment_y_start=-0.75,
    augment_y_end=0.75,
    n_augment_x=5,
    n_augment_y=5,
    trainedmodel_table=None,
    rename_data_key=True,
):

    if trainedmodel_table is None:
        trainedmodel_table = TrainedModel
    elif trainedmodel_table == "TrainedEnsembleModel":
        trainedmodel_table = TrainedEnsembleModel

    dataloaders, models = trainedmodel_table().load_model(key)

    n_models = len(models.members) if hasattr(models, "members") else 1

    for i in range(n_models):

        model = models.members[i] if hasattr(models, "members") else models
        data_key = list(model.readout.keys())[0]

        grid_augment = []
        for x in np.linspace(augment_x_start, augment_x_end, n_augment_x):
            for y in np.linspace(augment_y_start, augment_y_end, n_augment_y):
                grid_augment.append([x, y])
        grid_augment.append([0, 0])
        grid_augment = torch.tensor(grid_augment)
        neuron_repeats = grid_augment.shape[0]

        total_n_neurons = 0
        for data_key, readout in model.readout.items():
            if data_key == "augmentation":
                continue
            total_n_neurons += readout.outdims - (32 if mua_in else 0)

        n_augmented_units = total_n_neurons * neuron_repeats

        model.readout["augmentation"] = FullGaussian2d(
            in_shape=model.readout[data_key].in_shape,
            outdims=n_augmented_units,
            bias=True,
            gauss_type=model.readout[data_key].gauss_type,
        )
        insert_index = 0
        for data_key, readout in model.readout.items():

            if data_key == "augmentation":
                continue

            for i in range(readout.outdims - (32 if mua_in else 0)):
                features = model.readout[data_key].features.data[:, :, :, i]

                model.readout["augmentation"].features.data[
                    :, :, :, insert_index : insert_index + neuron_repeats
                ] = features[:, :, :, None]
                model.readout["augmentation"].bias.data[insert_index : insert_index + neuron_repeats] = model.readout[
                    data_key
                ].bias.data[i]
                model.readout["augmentation"].sigma.data[
                    :, insert_index : insert_index + neuron_repeats, :, :
                ] = model.readout[data_key].sigma.data[:, i, ...]
                model.readout["augmentation"].mu.data[
                    0, insert_index : insert_index + neuron_repeats, 0, :
                ] = grid_augment

                insert_index += neuron_repeats

        sessions = []
        for data_key in model.readout.keys():
            if data_key != "augmentation":
                sessions.append(data_key)

            if hasattr(model, "shifter"):
                pass

        for session in sessions:
            model.readout.pop(session)

        if rename_data_key is False:
            if len(sessions) > 1:
                raise ValueError("Renaming to original data key is only possible when dataloader has one data key only")
            model.readout[sessions[0]] = model.readout.pop("augmentation")

    return models
