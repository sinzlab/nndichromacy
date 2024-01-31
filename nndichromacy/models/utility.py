import torch
import copy
import math

def unpack_data_info(data_info):

    in_shapes_dict = {k: v["input_dimensions"] for k, v in data_info.items()}
    input_channels = [v["input_channels"] for k, v in data_info.items()]
    n_neurons_dict = {k: v["output_dimension"] for k, v in data_info.items()}
    return n_neurons_dict, in_shapes_dict, input_channels


def cart2pol_torch(x, y):
    """
    Change cartesian coordinates to polar
    """
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)
    return rho, phi


def unpack_data_info(data_info):

    in_shapes_dict = {k: v["input_dimensions"] for k, v in data_info.items()}
    input_channels = [v["input_channels"] for k, v in data_info.items()]
    n_neurons_dict = {k: v["output_dimension"] for k, v in data_info.items()}
    return n_neurons_dict, in_shapes_dict, input_channels


def purge_state_dict(state_dict, purge_key=None, survival_key=None):

    if (purge_key is None) and (survival_key is None):
        raise ValueError(
            "purge_key and survival_key can not both be None. At least one key has to be defined"
        )

    purged_state_dict = copy.deepcopy(state_dict)

    for dict_key in state_dict.keys():
        if (purge_key is not None) and (purge_key in dict_key):
            purged_state_dict.pop(dict_key)
        elif (survival_key is not None) and (survival_key not in dict_key):
            purged_state_dict.pop(dict_key)

    return purged_state_dict


def get_readout_key_names(model):
    data_key = list(model.readout.keys())[0]
    readout = model.readout[data_key]

    feature_name = "features"
    if "mu" in dir(readout):
        feature_name = "features"
        grid_name = "mu"
        bias_name = "bias"
    else:
        feature_name = "features"
        grid_name = "grid"
        bias_name = "bias"

    return feature_name, grid_name, bias_name


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe
