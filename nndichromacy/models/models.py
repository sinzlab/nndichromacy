import numpy as np
import torch
import copy

from mlutils.layers.cores import Stacked2dCore
from mlutils.layers.legacy import Gaussian2d
from mlutils.layers.readouts import PointPooled2d
from nnfabrik.models.pretrained_models import TransferLearningCore
from nnfabrik.utility.nn_helpers import get_module_output, set_random_seed, get_dims_for_loader_dict
from torch import nn
from torch.nn import functional as F

from .cores import SE2dCore, TransferLearningCore
from .readouts import MultipleFullGaussian2d, MultiReadout, MultipleSpatialXFeatureLinear
from .utility import unpack_data_info

class MultiplePointPooled2d(MultiReadout, torch.nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, pool_steps, pool_kern, bias, init_range, gamma_readout):
        # super init to get the _module attribute
        super(MultiplePointPooled2d, self).__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(k, PointPooled2d(
                in_shape,
                n_neurons,
                pool_steps=pool_steps,
                pool_kern=pool_kern,
                bias=bias,
                init_range=init_range)
                            )
        self.gamma_readout = gamma_readout


class MultipleGaussian2d(torch.nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, init_mu_range, init_sigma_range, bias, gamma_readout):
        # super init to get the _module attribute
        super(MultipleGaussian2d, self).__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(k, Gaussian2d(
                in_shape=in_shape,
                outdims=n_neurons,
                init_mu_range=init_mu_range,
                init_sigma_range=init_sigma_range,
                bias=bias)
                            )
        self.gamma_readout = gamma_readout

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return self[data_key].feature_l1(average=False) * self.gamma_readout


def se_core_gauss_readout(dataloaders, seed, hidden_channels=32, input_kern=13,  # core args
                          hidden_kern=3, layers=3, gamma_input=15.5,
                          skip=0, final_nonlinearity=True, momentum=0.9,
                          pad_input=False, batch_norm=True, hidden_dilation=1,
                          laplace_padding=None, input_regularizer='LaplaceL2norm',
                          init_mu_range=0.2, init_sigma_range=0.5, readout_bias=True,  # readout args,
                          gamma_readout=4, elu_offset=0, stack=None, se_reduction=32, n_se_blocks=1,
                          depth_separable=False, linear=False, data_info=None,
                          ):
    """
    Model class of a stacked2dCore (from mlutils) and a pointpooled (spatial transformer) readout

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]

        all other args: See Documentation of Stacked2dCore in mlutils.layers.cores and
            PointPooled2D in mlutils.layers.readouts

    Returns: An initialized model which consists of model.core and model.readout
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

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

            sample = kwargs["sample"] if 'sample' in kwargs else None
            x = self.readout(x, data_key=data_key, sample=sample)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

    set_random_seed(seed)

    core = SE2dCore(input_channels=core_input_channels,
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
                    linear=linear)

    readout = MultipleGaussian2d(core, in_shape_dict=in_shapes_dict,
                                 n_neurons_dict=n_neurons_dict,
                                 init_mu_range=init_mu_range,
                                 bias=readout_bias,
                                 init_sigma_range=init_sigma_range,
                                 gamma_readout=gamma_readout)

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def se_core_full_gauss_readout(dataloaders, seed, hidden_channels=32, input_kern=13,  # core args
                               hidden_kern=3, layers=3, gamma_input=15.5,
                               skip=0, final_nonlinearity=True, momentum=0.9,
                               pad_input=False, batch_norm=True, hidden_dilation=1,
                               laplace_padding=None, input_regularizer='LaplaceL2norm',
                               init_mu_range=0.2, init_sigma=1., readout_bias=True,  # readout args,
                               gamma_readout=4, elu_offset=0, stack=None, se_reduction=32, n_se_blocks=1,
                               depth_separable=False, linear=False, gauss_type='full',
                               grid_mean_predictor=None, share_features=False, share_grid=False, data_info=None,
                               ):
    """
    Model class of a stacked2dCore (from mlutils) and a pointpooled (spatial transformer) readout

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
        all other args: See Documentation of Stacked2dCore in mlutils.layers.cores and
            PointPooled2D in mlutils.layers.readouts

    Returns: An initialized model which consists of model.core and model.readout
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = list(input_channels.values())[0] if isinstance(input_channels, dict) else input_channels[0]

    source_grids = None
    grid_mean_predictor_type = None
    if grid_mean_predictor is not None:
        grid_mean_predictor = copy.deepcopy(grid_mean_predictor)
        grid_mean_predictor_type = grid_mean_predictor.pop('type')
        if grid_mean_predictor_type == 'cortex':
            input_dim = grid_mean_predictor.pop('input_dimensions', 2)
            source_grids = {k: v.dataset.neurons.cell_motor_coordinates[:, :input_dim] for k, v in dataloaders.items()}
        elif grid_mean_predictor_type == 'shared':
            pass

    shared_match_ids = None
    if share_features or share_grid:
        shared_match_ids = {k: v.dataset.neurons.multi_match_id for k, v in dataloaders.items()}
        all_multi_unit_ids = set(np.hstack(shared_match_ids.values()))

        for match_id in shared_match_ids.values():
            assert len(set(match_id) & all_multi_unit_ids) == len(all_multi_unit_ids), \
                'All multi unit IDs must be present in all datasets'

    class Encoder(nn.Module):

        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)

            sample = kwargs["sample"] if 'sample' in kwargs else None
            x = self.readout(x, data_key=data_key, sample=sample)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

    set_random_seed(seed)

    core = SE2dCore(input_channels=core_input_channels,
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
                    linear=linear)

    readout = MultipleFullGaussian2d(core, in_shape_dict=in_shapes_dict,
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
                                     shared_match_ids=shared_match_ids
                                     )

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def se_core_point_readout(dataloaders, seed, hidden_channels=32, input_kern=13,  # core args
                          hidden_kern=3, layers=3, gamma_input=15.5,
                          skip=0, final_nonlinearity=True, momentum=0.9,
                          pad_input=False, batch_norm=True, hidden_dilation=1,
                          laplace_padding=None, input_regularizer='LaplaceL2norm',
                          pool_steps=2, pool_kern=3, init_range=0.2, readout_bias=True,  # readout args,
                          gamma_readout=4, elu_offset=0, stack=None, se_reduction=32, n_se_blocks=1,
                          depth_separable=False, linear=False, data_info=None,
                          ):
    """
    Model class of a stacked2dCore (from mlutils) and a pointpooled (spatial transformer) readout

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]

        all other args: See Documentation of Stacked2dCore in mlutils.layers.cores and
            PointPooled2D in mlutils.layers.readouts

    Returns: An initialized model which consists of model.core and model.readout
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

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

            x = self.readout(x, data_key=data_key)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

    set_random_seed(seed)

    core = SE2dCore(input_channels=core_input_channels,
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
                    linear=linear)

    readout = MultiplePointPooled2d(core, in_shape_dict=in_shapes_dict,
                                    n_neurons_dict=n_neurons_dict,
                                    pool_steps=pool_steps,
                                    pool_kern=pool_kern,
                                    bias=readout_bias,
                                    gamma_readout=gamma_readout,
                                    init_range=init_range)

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def stacked2d_core_gaussian_readout(dataloaders, seed, hidden_channels=32, input_kern=13,  # core args
                                    hidden_kern=3, layers=3, gamma_hidden=0, gamma_input=0.1,
                                    skip=0, final_nonlinearity=True, core_bias=False, momentum=0.9,
                                    pad_input=False, batch_norm=True, hidden_dilation=1,
                                    laplace_padding=None, input_regularizer='LaplaceL2norm',
                                    readout_bias=True, init_mu_range=0.2, init_sigma_range=0.5,  # readout args,
                                    gamma_readout=0.1, elu_offset=0, stack=None, isotropic=True, data_info=None,
                                    ):
    """
    Model class of a stacked2dCore (from mlutils) and a pointpooled (spatial transformer) readout

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]

        all other args: See Documentation of Stacked2dCore in mlutils.layers.cores and
            PointPooled2D in mlutils.layers.readouts

    Returns: An initialized model which consists of model.core and model.readout
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

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

    core = Stacked2dCore(input_channels=core_input_channels,
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
                         stack=stack)

    readout = MultipleFullGaussian2d(core, in_shape_dict=in_shapes_dict,
                                     n_neurons_dict=n_neurons_dict,
                                     init_mu_range=init_mu_range,
                                     init_sigma=init_sigma_range,
                                     bias=readout_bias,
                                     gamma_readout=gamma_readout,
                                     gauss_type=isotropic
                                     )

    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def vgg_core_gauss_readout(dataloaders, seed,
                           input_channels=1, tr_model_fn='vgg16',  # begin of core args
                           model_layer=11, momentum=0.1, final_batchnorm=True,
                           final_nonlinearity=True, bias=False,
                           init_mu_range=0.4, init_sigma_range=0.6, readout_bias=True,  # begin or readout args
                           gamma_readout=0.002, elu_offset=-1, gauss_type='uncorrelated', data_info=None):
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
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

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

    core = TransferLearningCore(input_channels=core_input_channels,
                                tr_model_fn=tr_model_fn,
                                model_layer=model_layer,
                                momentum=momentum,
                                final_batchnorm=final_batchnorm,
                                final_nonlinearity=final_nonlinearity,
                                bias=bias)

    readout = MultipleGaussian2d(core, in_shape_dict=in_shapes_dict,
                                     n_neurons_dict=n_neurons_dict,
                                     init_mu_range=init_mu_range,
                                     bias=readout_bias,
                                     gamma_readout=gamma_readout,
                                     init_sigma_range=init_sigma_range,
                                 )

    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def vgg_core_full_gauss_readout(dataloaders, seed,
                           input_channels=1, tr_model_fn='vgg16',  # begin of core args
                           model_layer=11, momentum=0.1, final_batchnorm=True,
                           final_nonlinearity=True, bias=False,
                           init_mu_range=0.4, init_sigma_range=0.6, readout_bias=True,  # begin or readout args
                           gamma_readout=0.002, elu_offset=-1, gauss_type='uncorrelated', data_info=None):
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
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

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

    core = TransferLearningCore(input_channels=core_input_channels,
                                tr_model_fn=tr_model_fn,
                                model_layer=model_layer,
                                momentum=momentum,
                                final_batchnorm=final_batchnorm,
                                final_nonlinearity=final_nonlinearity,
                                bias=bias)


    readout = MultipleGaussian2d(core, in_shape_dict=in_shapes_dict,
                                 n_neurons_dict=n_neurons_dict,
                                 init_mu_range=init_mu_range,
                                 bias=readout_bias,
                                 init_sigma_range=init_sigma_range,
                                 gamma_readout=gamma_readout)
    
    
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def se_core_spatialXfeature_readout(dataloaders, seed, hidden_channels=32, input_kern=13,  # core args
                                    hidden_kern=3, layers=3, gamma_input=15.5,
                                    skip=0, final_nonlinearity=True, momentum=0.9,
                                    pad_input=False, batch_norm=True, hidden_dilation=1,
                                    laplace_padding=None, input_regularizer='LaplaceL2norm',
                                    init_noise=1e-3, readout_bias=True,  # readout args,
                                    gamma_readout=4, normalize=False, elu_offset=0, stack=None, se_reduction=32, n_se_blocks=1,
                                    depth_separable=False, linear=False, data_info=None):
    """
    Model class of a stacked2dCore (from mlutils) and a spatialXfeature (factorized) readout

    Args:

    Returns: An initialized model which consists of model.core and model.readout
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]


        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

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

            x = self.readout(x, data_key=data_key,)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

    set_random_seed(seed)

    core = SE2dCore(input_channels=core_input_channels,
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
                    linear=linear)

    readout = MultipleSpatialXFeatureLinear(core, in_shape_dict=in_shapes_dict,
                                            n_neurons_dict=n_neurons_dict,
                                            init_noise=init_noise,
                                            bias=readout_bias,
                                            gamma_readout=gamma_readout,
                                            normalize=normalize
                                            )

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model
