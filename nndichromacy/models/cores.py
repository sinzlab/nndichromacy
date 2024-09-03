from collections import OrderedDict
from collections.abc import Iterable
import torch
from torch import nn as nn
from torch.autograd import Variable
import numpy as np


from neuralpredictors.layers.cores import DepthSeparableConv2d, Core2d, Stacked2dCore
from neuralpredictors import regularizers
from neuralpredictors.regularizers import Laplace
from neuralpredictors.layers.attention import AttentionConv

from .utility import *
from .architectures import HermiteConv2D
from .architectures import SQ_EX_Block


try:
    from ptrnets import vgg19_original, vgg19_norm
except:
    pass


class TransferLearningCore(Core2d, nn.Module):
    """
    A Class to create a Core based on a model class from torchvision.models.
    """

    def __init__(
        self,
        input_channels,
        tr_model_fn,
        model_layer,
        pretrained=True,
        final_batchnorm=True,
        final_nonlinearity=True,
        bias=False,
        momentum=0.1,
        fine_tune=False,
        **kwargs
    ):
        """
        Args:
            input_channels: number of input channgels
            tr_model_fn: string to specify the pretrained model, as in torchvision.models, e.g. 'vgg16'
            model_layer: up onto which layer should the pretrained model be built
            pretrained: boolean, if pretrained weights should be used
            final_batchnorm: adds a batch norm layer
            final_nonlinearity: adds a nonlinearity
            bias: Adds a bias term.
            momentum: batch norm momentum
            fine_tune: boolean, sets all weights to trainable if True
            **kwargs:
        """
        print(
            "Ignoring input {} when creating {}".format(
                repr(kwargs), self.__class__.__name__
            )
        )
        super().__init__()

        # getattr(self, tr_model_fn)
        tr_model_fn = globals()[tr_model_fn]

        self.input_channels = input_channels
        self.tr_model_fn = tr_model_fn

        tr_model = tr_model_fn(pretrained=pretrained)
        self.model_layer = model_layer
        self.features = nn.Sequential()

        tr_features = nn.Sequential(*list(tr_model.features.children())[:model_layer])

        # Remove the bias of the last conv layer if not :bias:
        if not bias:
            if "bias" in tr_features[-1]._parameters:
                zeros = torch.zeros_like(tr_features[-1].bias)
                tr_features[-1].bias.data = zeros

        # Fix pretrained parameters during training parameters
        if not fine_tune:
            for param in tr_features.parameters():
                param.requires_grad = False

        self.features.add_module("TransferLearning", tr_features)
        print(self.features)
        if final_batchnorm:
            self.features.add_module(
                "OutBatchNorm", nn.BatchNorm2d(self.outchannels, momentum=momentum)
            )
        if final_nonlinearity:
            self.features.add_module("OutNonlin", nn.ReLU(inplace=True))

    def forward(self, x):
        if self.input_channels == 1:
            x = x.expand(-1, 3, -1, -1)
        return self.features(x)

    def regularizer(self):
        return 0

    @property
    def outchannels(self):
        """
        Returns: dimensions of the output, after a forward pass through the model
        """
        found_out_channels = False
        i = 1
        while not found_out_channels:
            if "out_channels" in self.features.TransferLearning[-i].__dict__:
                found_out_channels = True
            else:
                i = i + 1
        return self.features.TransferLearning[-i].out_channels


class SE2dCore(Core2d, nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        input_kern,
        hidden_kern,
        layers=3,
        gamma_input=0.0,
        skip=0,
        final_nonlinearity=True,
        bias=False,
        momentum=0.1,
        pad_input=True,
        batch_norm=True,
        hidden_dilation=0,
        laplace_padding=None,
        input_regularizer="LaplaceL2norm",
        stack=None,
        se_reduction=32,
        n_se_blocks=1,
        depth_separable=False,
        attention_conv=False,
        linear=False,
        hidden_padding=None,
    ):
        """
        Args:
            input_channels:     Integer, number of input channels as in
            hidden_channels:    Number of hidden channels (i.e feature maps) in each hidden layer
            input_kern:     kernel size of the first layer (i.e. the input layer)
            hidden_kern:    kernel size of each hidden layer's kernel
            layers:         number of layers
            gamma_input:    regularizer factor for the input weights (default: LaplaceL2, see neuralpredictors.regularizers)
            skip:           Adds a skip connection
            final_nonlinearity: Boolean, if true, appends an ELU layer after the last BatchNorm (if BN=True)
            bias:           Adds a bias layer. Note: bias and batch_norm can not both be true
            momentum:       BN momentum
            pad_input:      Boolean, if True, applies zero padding to all convolutions
            batch_norm:     Boolean, if True appends a BN layer after each convolutional layer
            hidden_dilation:    If set to > 1, will apply dilated convs for all hidden layers
            laplace_padding: Padding size for the laplace convolution. If padding = None, it defaults to half of
                the kernel size (recommended). Setting Padding to 0 is not recommended and leads to artefacts,
                zero is the default however to recreate backwards compatibility.
            normalize_laplace_regularizer: Boolean, if set to True, will use the LaplaceL2norm function from
                neuralpredictors.regularizers, which returns the regularizer as |laplace(filters)| / |filters|
            input_regularizer:  String that must match one of the regularizers in ..regularizers
            stack:        Int or iterable. Selects which layers of the core should be stacked for the readout.
                            default value will stack all layers on top of each other.
                            stack = -1 will only select the last layer as the readout layer
                            stack = 0  will only readout from the first layer
            se_reduction:   Int. Reduction of Channels for Global Pooling of the Squeeze and Excitation Block.
            attention_conv: Boolean, if True, uses self-attention instead of convolution for layers 2 and following
        """

        super().__init__()

        assert not bias or not batch_norm, "bias and batch_norm should not both be true"
        assert (
            not depth_separable or not attention_conv
        ), "depth_separable and attention_conv should not both be true"

        regularizer_config = (
            dict(padding=laplace_padding, kernel=input_kern)
            if input_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )
        self._input_weights_regularizer = regularizers.__dict__[input_regularizer](
            **regularizer_config
        )

        self.layers = layers
        self.gamma_input = gamma_input
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.skip = skip
        self.features = nn.Sequential()
        self.n_se_blocks = n_se_blocks
        if stack is None:
            self.stack = range(self.layers)
        else:
            self.stack = (
                [*range(self.layers)[stack:]] if isinstance(stack, int) else stack
            )

        # --- first layer
        layer = OrderedDict()
        layer["conv"] = nn.Conv2d(
            input_channels,
            hidden_channels,
            input_kern,
            padding=input_kern // 2 if pad_input else 0,
            bias=bias,
        )
        if batch_norm:
            layer["norm"] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
        if (layers > 1 or final_nonlinearity) and not linear:
            layer["nonlin"] = nn.ELU(inplace=True)
        self.features.add_module("layer0", nn.Sequential(layer))

        if not isinstance(hidden_kern, Iterable):
            hidden_kern = [hidden_kern] * (self.layers - 1)

        # --- other layers
        for l in range(1, self.layers):
            layer = OrderedDict()
            hidden_padding = (
                hidden_padding
                if hidden_padding is not None
                else ((hidden_kern[l - 1] - 1) * hidden_dilation + 1) // 2
            )
            if depth_separable:
                layer["ds_conv"] = DepthSeparableConv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=hidden_kern[l - 1],
                    dilation=hidden_dilation,
                    padding=hidden_padding,
                    bias=False,
                    stride=1,
                )
            elif attention_conv:
                layer["conv"] = AttentionConv(
                    hidden_channels if not skip > 1 else min(skip, l) * hidden_channels,
                    hidden_channels,
                    hidden_kern[l - 1],
                    padding=hidden_padding,
                    bias=bias and not batch_norm,
                )
            else:
                layer["conv"] = nn.Conv2d(
                    hidden_channels if not skip > 1 else min(skip, l) * hidden_channels,
                    hidden_channels,
                    hidden_kern[l - 1],
                    padding=hidden_padding,
                    bias=bias,
                    dilation=hidden_dilation,
                )
            if batch_norm:
                layer["norm"] = nn.BatchNorm2d(hidden_channels, momentum=momentum)

            if (final_nonlinearity or l < self.layers - 1) and not linear:
                layer["nonlin"] = nn.ELU(inplace=True)

            if (self.layers - l) <= self.n_se_blocks:
                layer["seg_ex_block"] = SQ_EX_Block(
                    in_ch=hidden_channels, reduction=se_reduction
                )

            self.features.add_module("layer{}".format(l), nn.Sequential(layer))

        self.apply(self.init_conv)

    def forward(self, input_):
        ret = []
        for l, feat in enumerate(self.features):
            do_skip = l >= 1 and self.skip > 1
            input_ = feat(
                input_ if not do_skip else torch.cat(ret[-min(self.skip, l) :], dim=1)
            )
            if l in self.stack:
                ret.append(input_)
        return torch.cat(ret, dim=1)

    def laplace(self):
        return self._input_weights_regularizer(self.features[0].conv.weight)

    def regularizer(self):
        return self.gamma_input * self.laplace()

    @property
    def outchannels(self):
        return len(self.features) * self.hidden_channels


class RotationEquivariantCore(nn.Module):
    def __init__(
        self,
        num_rotations=8,
        upsampling=2,
        filter_size=[13, 5, 5],
        num_filters=[8, 16, 32],
    ):

        super(RotationEquivariantCore, self).__init__()
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_rotations = num_rotations

        layers = []
        input_features = 1
        for i, (fs, nf) in enumerate(zip(filter_size, num_filters)):
            layer = HermiteConv2D(
                input_features=input_features,
                output_features=nf,
                filter_size=fs,
                padding=0,
                stride=1,
                num_rotations=num_rotations,
                upsampling=upsampling,
                first_layer=(not i),
                layer_id=i,
            )
            layers.append(layer)

            input_features = nf * num_rotations

        self.layers = nn.Sequential(*layers)

    def regularizer(self):
        return 0

    def forward(self, input):
        conv = input
        for layer in self.layers:
            conv = layer(conv)
        return conv


class LaplaceL2(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self, padding=0):
        super().__init__()
        self.laplace = Laplace(padding=padding)

    def forward(self, x, weights=None):
        ic, oc, k1, k2 = x.size()
        if weights is None:
            weights = 1.0
        return (self.laplace(x.view(ic * oc, 1, k1, k2)).view(ic, oc, k1, k2).pow(2) * weights).mean() / 2

#### code from cajal/static-networks/staticnet.cores.py
class StaticnetStacked2dCore(Core2d, nn.Module):
    def __init__(self, input_channels, hidden_channels, input_kern, hidden_kern, layers=3,
                 gamma_hidden=0, gamma_input=0., skip=0, final_nonlinearity=True, bias=False, skip_nonlin=False,
                 momentum=0.1, pad_input=True, batch_norm=True, laplace_padding=0, laplace_weights_fn=None, **kwargs):
        #         log.info('Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__))
        super().__init__()

        if skip_nonlin:
            print('Skip non-linearity')

        self._input_weights_regularizer = LaplaceL2(padding=laplace_padding)

        self.layers = layers
        self.gamma_input = gamma_input
        self.gamma_hidden = gamma_hidden
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.skip = skip

        self.features = nn.Sequential()
        # --- first layer
        layer = OrderedDict()
        layer['conv'] = nn.Conv2d(input_channels, hidden_channels, input_kern, padding=input_kern // 2 if pad_input else 0, bias=bias)

        if batch_norm:
            layer['norm'] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
        if (not skip_nonlin) and (layers > 1 or final_nonlinearity):
            layer['nonlin'] = nn.ELU(inplace=True)
        self.features.add_module('layer0', nn.Sequential(layer))

        # --- other layers
        for l in range(1, self.layers):
            layer = OrderedDict()
            layer['conv'] = \
                nn.Conv2d(hidden_channels if not skip > 1 else min(skip, l) * hidden_channels,
                          hidden_channels, hidden_kern,
                          padding=hidden_kern // 2, bias=bias)

            if batch_norm:
                layer['norm'] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
            if (not skip_nonlin) and (final_nonlinearity or l < self.layers - 1):
                layer['nonlin'] = nn.ELU(inplace=True)
            self.features.add_module('layer{}'.format(l), nn.Sequential(layer))

        self.apply(self.init_conv)

        if laplace_weights_fn is not None:
            _, _, h, w = self.features[0].conv.weight.size()
            dist_grid = torch.sqrt(torch.linspace(-1, 1, h)[:, None].pow(2) + torch.linspace(-1, 1, w).pow(2))
            self.register_buffer('laplace_weights', laplace_weights_fn(dist_grid))
        else:
            self.laplace_weights = None


    def forward(self, input_):
        ret = []
        for l, feat in enumerate(self.features):
            do_skip = l >= 1 and self.skip > 1
            input_ = feat(input_ if not do_skip else torch.cat(ret[-min(self.skip, l):], dim=1))
            ret.append(input_)
        return torch.cat(ret, dim=1)

    def laplace(self):
        return self._input_weights_regularizer(self.features[0].conv.weight, weights=self.laplace_weights)

    def group_sparsity(self):
        ret = 0
        for l in range(1, self.layers):
            ret = ret + self.features[l].conv.weight.pow(2).sum(3, keepdim=True).sum(2, keepdim=True).sqrt().mean()
        return ret / ((self.layers - 1) if self.layers > 1 else 1)

    def regularizer(self):
        return self.group_sparsity() * self.gamma_hidden + self.gamma_input * self.laplace()

    @staticmethod
    def get_readout_in_shape(core, in_shape):
        mov_shape = in_shape[1:]
        core.eval()
        tmp = Variable(torch.from_numpy(np.random.randn(1, *mov_shape).astype(np.float32)))
        nout = core(tmp).size()[1:]
        core.train(True)
        return nout

    @property
    def outchannels(self):
        return len(self.features) * self.hidden_channels

class GaussianLaplaceCore(StaticnetStacked2dCore):
    def __init__(self, gauss_sigma, gauss_bias, *args,  **kwargs):
        laplace_weights_fn = lambda x: 1 - torch.exp(-x.pow(2) / 2 / gauss_sigma**2) + gauss_bias
        super().__init__(*args, laplace_weights_fn=laplace_weights_fn, **kwargs)