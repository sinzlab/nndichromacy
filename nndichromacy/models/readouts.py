import numpy as np
import torch
from torch import nn as nn
from torch.nn import Parameter
from torch.nn import functional as F

from neuralpredictors.utils import get_module_output
from neuralpredictors.layers.readouts import (
    PointPooled2d,
    FullGaussian2d,
    SpatialXFeatureLinear,
    PointPyramid2d,
)
from neuralpredictors.layers.legacy import Gaussian2d
import torch
from torch import nn
from neuralpredictors.constraints import positive
from .utility import cart2pol_torch


class MultiplePointPooled2d(torch.nn.ModuleDict):
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

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return self[data_key].feature_l1(average=False) * self.gamma_readout


class MultiplePointPyramid2d(torch.nn.ModuleDict):
    def __init__(
            self,
            core,
            in_shape_dict,
            n_neurons_dict,
            gamma_readout=1,
            scale_n=5,
            positive=False,
            bias=True,
            init_range=.1,
            downsample=False,
            type='gauss5x5',
            align_corners=True,
    ):
        # super init to get the _module attribute
        super(MultiplePointPyramid2d, self).__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(
                k,
                PointPyramid2d(
                    in_shape,
                    n_neurons,
                    scale_n=scale_n,
                    positive=positive,
                    bias=bias,
                    init_range=init_range,
                    downsample=downsample,
                    type=type,
                    align_corners=align_corners,
                ),
            )
        self.gamma_readout = gamma_readout

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return self[data_key].feature_l1(average=False) * self.gamma_readout


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


class MultiReadout:
    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return self[data_key].feature_l1(average=False) * self.gamma_readout


class MultipleSpatialXFeatureLinear(MultiReadout, torch.nn.ModuleDict):
    def __init__(
        self,
        core,
        in_shape_dict,
        n_neurons_dict,
        init_noise,
        bias,
        normalize,
        gamma_readout,
    ):
        # super init to get the _module attribute
        super().__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(
                k,
                SpatialXFeatureLinear(
                    in_shape=in_shape,
                    outdims=n_neurons,
                    init_noise=init_noise,
                    bias=bias,
                    normalize=normalize,
                ),
            )
        self.gamma_readout = gamma_readout

    def regularizer(self, data_key):
        return self[data_key].l1(average=False) * self.gamma_readout


class MultipleFullGaussian2d(MultiReadout, torch.nn.ModuleDict):
    def __init__(
        self,
        core,
        in_shape_dict,
        n_neurons_dict,
        init_mu_range,
        init_sigma,
        bias,
        gamma_readout,
        gauss_type,
        grid_mean_predictor,
        grid_mean_predictor_type,
        source_grids,
        share_features,
        share_grid,
        shared_match_ids,
    ):
        # super init to get the _module attribute
        super().__init__()
        k0 = None
        for i, k in enumerate(n_neurons_dict):
            k0 = k0 or k
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]

            source_grid = None
            shared_grid = None
            shared_transform = None
            if grid_mean_predictor is not None:
                if grid_mean_predictor_type == "cortex":
                    source_grid = source_grids[k]
                else:
                    raise KeyError(
                        "grid mean predictor {} does not exist".format(
                            grid_mean_predictor_type
                        )
                    )

            elif share_grid:
                shared_grid = {
                    "match_ids": shared_match_ids[k],
                    "shared_grid": None if i == 0 else self[k0].shared_grid,
                }

            if share_features:
                shared_features = {
                    "match_ids": shared_match_ids[k],
                    "shared_features": None if i == 0 else self[k0].shared_features,
                }
            else:
                shared_features = None

            self.add_module(
                k,
                FullGaussian2d(
                    in_shape=in_shape,
                    outdims=n_neurons,
                    init_mu_range=init_mu_range,
                    init_sigma=init_sigma,
                    bias=bias,
                    gauss_type=gauss_type,
                    grid_mean_predictor=grid_mean_predictor,
                    shared_features=shared_features,
                    shared_grid=shared_grid,
                    source_grid=source_grid,
                ),
            )
            
        self.gamma_readout = gamma_readout


class FullGaussBehavior(FullGaussian2d):
    """
    A readout using a spatial transformer layer whose positions are sampled from one Gaussian per neuron. Mean
    and covariance of that Gaussian are learned.

    Args:
        in_shape (list, tuple): shape of the input feature map [channels, width, height]
        outdims (int): number of output units
        bias (bool): adds a bias term
        behavioral_channels (int): The number of behavioral channels.
        init_mu_range (float): initialises the the mean with Uniform([-init_range, init_range])
                            [expected: positive value <=1]. Default: 0.1
        init_sigma (float): The standard deviation of the Gaussian with `init_sigma` when `gauss_type` is
            'isotropic' or 'uncorrelated'. When `gauss_type='full'` initialize the square root of the
            covariance matrix with with Uniform([-init_sigma, init_sigma]). Default: 1
        batch_sample (bool): if True, samples a position for each image in the batch separately
                            [default: True as it decreases convergence time and performs just as well]
        align_corners (bool): Keyword agrument to gridsample for bilinear interpolation.
                It changed behavior in PyTorch 1.3. The default of align_corners = True is setting the
                behavior to pre PyTorch 1.3 functionality for comparability.
        gauss_type (str): Which Gaussian to use. Options are 'isotropic', 'uncorrelated', or 'full' (default).
        grid_mean_predictor (dict): Parameters for a predictor of the mean grid locations. Has to have a form like
                        {
                        'hidden_layers':0,
                        'hidden_features':20,
                        'final_tanh': False,
                        }
        shared_features (dict): Used when the feature vectors are shared (within readout between neurons) or between
                this readout and other readouts. Has to be a dictionary of the form
               {
                    'match_ids': (numpy.array),
                    'shared_features': torch.nn.Parameter or None
                }
                The match_ids are used to match things that should be shared within or across scans.
                If `shared_features` is None, this readout will create its own features. If it is set to
                a feature Parameter of another readout, it will replace the features of this readout. It will be
                access in increasing order of the sorted unique match_ids. For instance, if match_ids=[2,0,0,1],
                there should be 3 features in order [0,1,2]. When this readout creates features, it will do so in
                that order.
        shared_grid (dict): Like `shared_features`. Use dictionary like
               {
                    'match_ids': (numpy.array),
                    'shared_grid': torch.nn.Parameter or None
                }
                See documentation of `shared_features` for specification.

        source_grid (numpy.array):
                Source grid for the grid_mean_predictor.
                Needs to be of size neurons x grid_mean_predictor[input_dimensions]

    """

    def __init__(
        self,
        *args,
        behavior_channels,
        **kwargs,
    ):
        self.behavior_channels = behavior_channels
        super().__init__(*args, **kwargs)
        self.initialize_modulator()

    def initialize_modulator(self):
        self.modulator = nn.Linear(
            in_features=self.behavior_channels, out_features=self.outdims, bias=False
        )
        self.modulator.weight.data.fill_(1 / self.behavior_channels)

    def initialize_features(self, match_ids=None, shared_features=None):
        """
        The internal attribute `_original_features` in this function denotes whether this instance of the FullGuassian2d
        learns the original features (True) or if it uses a copy of the features from another instance of FullGaussian2d
        via the `shared_features` (False). If it uses a copy, the feature_l1 regularizer for this copy will return 0
        """
        c, w, h = self.in_shape
        self._original_features = True
        if match_ids is not None:
            assert self.outdims == len(match_ids)

            n_match_ids = len(np.unique(match_ids))
            if shared_features is not None:
                assert shared_features.shape == (
                    1,
                    c,
                    1,
                    n_match_ids,
                ), f"shared features need to have shape (1, {c}, 1, {n_match_ids})"
                self._features = shared_features
                self._original_features = False
            else:
                self._features = Parameter(
                    torch.Tensor(1, c, 2, n_match_ids)
                )  # feature weights for each channel of the core
            self.scales = Parameter(
                torch.Tensor(1, 1, 1, self.outdims)
            )  # feature weights for each channel of the core
            _, sharing_idx = np.unique(match_ids, return_inverse=True)
            self.register_buffer("feature_sharing_index", torch.from_numpy(sharing_idx))
            self._shared_features = True
        else:
            self._features = Parameter(
                torch.Tensor(1, c, 2, self.outdims)
            )  # feature weights for each channel of the core
            self._shared_features = False

    def initialize(self):
        """
        Initializes the mean, and sigma of the Gaussian readout along with the features weights
        """

        if not self._predicted_grid or self._original_grid:
            self._mu.data.uniform_(-self.init_mu_range, self.init_mu_range)

        if self.gauss_type != "full":
            self.sigma.data.fill_(self.init_sigma)
        else:
            self.sigma.data.uniform_(-self.init_sigma, self.init_sigma)
        self._features.data.fill_(1 / self.in_shape[0])
        if self._shared_features:
            self.scales.data.fill_(1.0)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def modulator_l1(self):
        return self.modulator.weight.sum()

    def feature_dissimilarity_l1(self):
        return -(self._features[:, :, 0, :] - self._features[:, :, 1, :]).abs().sum()

    def forward(self, x, behavior, sample=None, shift=None, out_idx=None, **kwargs):
        """
        Propagates the input forwards through the readout
        Args:
            x: input data
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
            shift (bool): shifts the location of the grid (from eye-tracking data)
            out_idx (bool): index of neurons to be predicted

        Returns:
            y: neuronal activity
        """
        N, c, w, h = x.size()
        c_in, w_in, h_in = self.in_shape
        if (c_in, w_in, h_in) != (c, w, h):
            raise ValueError(
                "the specified feature map dimension is not the readout's expected input dimension"
            )
        feat = self.features.view(2, c, self.outdims)
        bias = self.bias
        outdims = self.outdims

        if self.batch_sample:
            # sample the grid_locations separately per image per batch
            grid = self.sample_grid(
                batch_size=N, sample=sample
            )  # sample determines sampling from Gaussian
        else:
            # use one sampled grid_locations for all images in the batch
            grid = self.sample_grid(batch_size=1, sample=sample).expand(
                N, outdims, 1, 2
            )

        if out_idx is not None:
            if isinstance(out_idx, np.ndarray):
                if out_idx.dtype == bool:
                    out_idx = np.where(out_idx)[0]
            feat = feat[:, :, out_idx]
            grid = grid[:, out_idx]
            if bias is not None:
                bias = bias[out_idx]
            outdims = len(out_idx)

        if shift is not None:
            grid = grid + shift[:, None, None, :]
        y = F.grid_sample(x, grid, align_corners=self.align_corners)

        lambda_behav = torch.sigmoid(self.modulator(behavior))

        f1 = torch.einsum("bn,cn->bcn", lambda_behav, feat[0])
        f2 = torch.einsum("bn,cn->bcn", (1 - lambda_behav), feat[1])
        y = (y.squeeze(-1) * (f1 + f2)).sum(1).view(N, outdims)

        if self.bias is not None:
            y = y + bias
        return y


class MultipleFullGaussian2dBehav(MultiReadout, torch.nn.ModuleDict):
    def __init__(
        self,
        core,
        in_shape_dict,
        n_neurons_dict,
        init_mu_range,
        init_sigma,
        bias,
        gamma_readout,
        gauss_type,
        behavior_channels,
        grid_mean_predictor,
        grid_mean_predictor_type,
        source_grids,
        share_features,
        share_grid,
        shared_match_ids,
        gamma_modulator=0,
        gamma_dissimilarity=0,
    ):
        # super init to get the _module attribute
        super().__init__()
        k0 = None
        for i, k in enumerate(n_neurons_dict):
            k0 = k0 or k
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]

            source_grid = None
            shared_grid = None
            shared_transform = None
            if grid_mean_predictor is not None:
                if grid_mean_predictor_type == "cortex":
                    source_grid = source_grids[k]
                else:
                    raise KeyError(
                        "grid mean predictor {} does not exist".format(
                            grid_mean_predictor_type
                        )
                    )

            elif share_grid:
                shared_grid = {
                    "match_ids": shared_match_ids[k],
                    "shared_grid": None if i == 0 else self[k0].shared_grid,
                }

            if share_features:
                shared_features = {
                    "match_ids": shared_match_ids[k],
                    "shared_features": None if i == 0 else self[k0].shared_features,
                }
            else:
                shared_features = None

            self.add_module(
                k,
                FullGaussBehavior(
                    in_shape=in_shape,
                    outdims=n_neurons,
                    init_mu_range=init_mu_range,
                    init_sigma=init_sigma,
                    bias=bias,
                    behavior_channels=behavior_channels,
                    gauss_type=gauss_type,
                    grid_mean_predictor=grid_mean_predictor,
                    shared_features=shared_features,
                    shared_grid=shared_grid,
                    source_grid=source_grid,
                ),
            )
        self.gamma_readout = gamma_readout
        self.gamma_modulator = gamma_modulator if gamma_modulator is not None else 0
        self.gamma_dissimilarity = (
            gamma_dissimilarity if gamma_dissimilarity is not None else 0
        )

    def regularizer(self, data_key):
        return (
            self[data_key].feature_l1(average=False) * self.gamma_readout
            + self[data_key].modulator_l1() * self.gamma_modulator
            + self[data_key].feature_dissimilarity_l1() * self.gamma_dissimilarity
        )


class CenterSurround2d(nn.Module):

    def __init__(
            self,
            in_shape,
            outdims,
            bias,
            init_mu_range=0.1,
            init_center_var=.02,
            init_surround_var=.02,
            init_surround_radius=.45,
            center_on=True,
            surround_on=True,
    ):

        super().__init__()

        # make sure the inputs are withing the proper range
        if init_mu_range > 1.0 or init_mu_range <= 0.0 or init_center_var <= 0.0 or init_surround_var <= 0.0:
            raise ValueError("either init_mu_range doesn't belong to [0.0, 1.0] or init_var is non-positive")


        self.in_shape = in_shape
        c, w, h = in_shape
        self.outdims = outdims
        self.init_mu_range = init_mu_range
        self.init_center_var = init_center_var
        self.init_surround_var = init_surround_var
        self.init_surround_radius = init_surround_radius
        self.center_on = center_on
        self.surround_on = surround_on
        self.detach_center = False

        # shared (between center and surround) params
        self._mu = nn.Parameter(data=torch.zeros(outdims, 2), requires_grad=True)

        # center params
        self._center_var = nn.Parameter(torch.zeros(outdims), requires_grad=True)
        self._center_weights = nn.Parameter(torch.Tensor(outdims), requires_grad=True)
        self._center_feature_weights = nn.Parameter(torch.Tensor(c, outdims))

        # surround params
        self._surround_var = nn.Parameter(torch.zeros(outdims), requires_grad=True)
        self._surround_radius = nn.Parameter(torch.zeros(outdims), requires_grad=True)
        self._surround_weights = nn.Parameter(torch.Tensor(outdims), requires_grad=True)
        self._surround_feature_weights = nn.Parameter(torch.Tensor(c, outdims))

        # other params
        self.grid = nn.Parameter(data=self.make_mask_grid(), requires_grad=False)
        self.bias = nn.Parameter(torch.Tensor(outdims))


        self.initialize()
        self.record_initial_values()

    def initialize(self):

        self._mu.data.uniform_(-self.init_mu_range, self.init_mu_range)

        self._center_var.data.fill_(self.init_center_var)
        self._center_weights.data.fill_(1.)
        self._center_feature_weights.data.fill_(1 / self.in_shape[0])

        self._surround_var.data.fill_(self.init_surround_var)
        self._surround_radius.data.fill_(self.init_surround_radius)
        self._surround_weights.data.fill_(0.)
        self._surround_feature_weights.data.fill_(1 / self.in_shape[0])

        if self.bias is None:
            self.bias.data.fill_(0)
            self.bias.requires_grad_(False)

        else:
            self.bias.data.fill_(0)

    def record_initial_values(self):
        self.center_weights_init = self.center_weights.detach()
        self.surround_weights_init = self.surround_weights.detach()

    def make_mask_grid(self):
        xx, yy = torch.meshgrid(
            [
                torch.linspace(-1, 1, self.in_shape[1]),
                torch.linspace(-1, 1, self.in_shape[2]),
            ]
        )
        grid = torch.stack([xx, yy], 2)[None, ...]
        return grid.repeat([self.outdims, 1, 1, 1])

    @property
    def mu(self):
        self._mu.data.clamp_(-1, 1)
        return self._mu

    @property
    def center_var(self):
        self._center_var.data.clamp_(1e-3, .05)
        return self._center_var

    @property
    def surround_var(self):
        self._surround_var.data.clamp_(1e-3, .03)
        return self._surround_var

    @property
    def surround_radius(self):
        surround_radius = torch.max(self._surround_radius, (3 * self.center_var.sqrt()/2 + 3 * self.surround_var.sqrt()/2).data)
        surround_radius = torch.min(surround_radius, 1. - 3 * self.surround_var.sqrt()/2)
        return surround_radius

    @property
    def center_weights(self):
        return self._center_weights

    @property
    def surround_weights(self):
        return self._surround_weights

    @property
    def center_feature_weights(self):
        # make the feature weights for each neuron positive and unit length
        positive(self._center_feature_weights)
        return self._center_feature_weights / torch.norm(self._center_feature_weights, p=2, dim=0, keepdim=True)

    @property
    def surround_feature_weights(self):
        # make the feature weights for each neuron positive and unit length
        positive(self._surround_feature_weights)
        return self._surround_feature_weights / torch.norm(self._surround_feature_weights, p=2, dim=0, keepdim=True)

    def mask(self, shift=None):

        if shift is None:
            mu = self.mu
        else:
            mu = self.mu + shift[None, ...]
        mean = mu.view(self.outdims, 1, 1, -1)

        # center mask
        center_variances = self.center_var.view(-1, 1, 1)
        center_pdf = self.grid - mean
        center_pdf = torch.sum(center_pdf ** 2, dim=-1) / center_variances
        center_pdf = torch.exp(-0.5 * center_pdf)
        # normalize to sum=1
        center_pdf = center_pdf / torch.sum(center_pdf, dim=(1, 2), keepdim=True)

        # surround mask
        surround_variances = self.surround_var.view(-1, 1, 1)
        rho, phi = cart2pol_torch((self.grid - mean)[:, :, :, 0], (self.grid - mean)[:, :, :, 1])
        radius = self.surround_radius.view(-1, 1, 1)
        surround_pdf = torch.exp(-.5 * (rho - radius) ** 2 / surround_variances)
        # normalize to sum=1
        surround_pdf = surround_pdf / torch.sum(surround_pdf, dim=(1, 2), keepdim=True)

        center_mask = center_pdf * self.center_weights.view(-1, 1, 1)
        surround_mask = surround_pdf * self.surround_weights.view(-1, 1, 1)

        return {"center": center_mask, "surround": surround_mask}


    def forward(self, center_features, surround_features, data_key=None, shift=None, **kwargs):
        Nc, cc, wc, hc = center_features.size()
        Ns, cs, ws, hs = surround_features.size()

        assert Nc == Ns, "batch size should be the same for center and surround features"

        center_feature_weights = self.center_feature_weights
        surround_feature_weights = self.surround_feature_weights

        mask = self.mask() if shift is None else self.mask(shift=shift)
        center_mask = mask["center"]
        surround_mask = mask["surround"]

        if self.center_on and self.surround_on:
            center_y = torch.einsum("bcij,nij,cn->bn", center_features, center_mask, center_feature_weights)
            if self.detach_center:
                center_y = center_y.detach()

            surround_y = torch.einsum("bcij,nij,cn->bn", surround_features, surround_mask, surround_feature_weights)
            y = center_y + surround_y

        elif self.center_on and not self.surround_on:
            center_y = torch.einsum("bcij,nij,cn->bn", center_features, center_mask, center_feature_weights)
            if self.detach_center:
                center_y = center_y.detach()
            y = center_y

        elif not self.center_on and self.surround_on:
            surround_y = torch.einsum("bcij,nij,cn->bn", surround_features, surround_mask, surround_feature_weights)
            y = surround_y

        return y + self.bias

    def feature_l1(self, average=True):
        """
        feature_l1 function returns the l1 regularization term either the mean or just the sum of weights
        Args:
            average(bool): if True, use mean of weights for regularization
        """
        raise NotImplementedError("self.features do not exist")
        #if average:
        #    return self.features.abs().mean()
        #else:
        #    return self.features.abs().sum()

    def regularizer(self, data_key=None):
        return 0.


class MultipleCenterSurround(MultiReadout, torch.nn.ModuleDict):
    def __init__(
            self,
            core,
            in_shape_dict,
            n_neurons_dict,
            gamma_readout,
            bias,
            init_mu_range=0.1,
            init_center_var=.02,
            init_surround_var=.02,
            init_surround_radius=.45,
            center_on=True,
            surround_on=True,
    ):
        # super init to get the _module attribute
        super().__init__()
        k0 = None
        for i, k in enumerate(n_neurons_dict):
            k0 = k0 or k
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]

            self.add_module(
                k,
                CenterSurround2d(
                    in_shape=in_shape,
                    outdims=n_neurons,
                    init_mu_range=init_mu_range,
                    init_center_var=init_center_var,
                    bias=bias,
                    init_surround_var=init_surround_var,
                    init_surround_radius=init_surround_radius,
                    center_on=center_on,
                    surround_on=surround_on,
                ),
            )
        self.gamma_readout = gamma_readout

    def regularizer(self, data_key):
        return self[data_key].regularizer()