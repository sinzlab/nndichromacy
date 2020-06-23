import torch

from nnfabrik.utility.nnf_helper import split_module_name, dynamic_import
from nnfabrik.utility.nn_helpers import get_dims_for_loader_dict
from . import core


def import_path(path):
    return dynamic_import(*split_module_name(path))


def gradient_ascent(
    dataloaders,
    model,
    config,
    import_object=import_path,
    get_dims=get_dims_for_loader_dict,
    get_initial_guess=torch.randn,
    ascend=core.gradient_ascent,
    initial_image=None,
):
    """Generates MEIs.

    Args:
        dataloaders: NNFabrik style dataloader dictionary.
        model: Callable object that returns a single number.
        config: Dictionary of arguments for the gradient ascent function.
        import_object: Function used to import functions given a path as a string. For testing purposes.
        get_dims: Function used to get the input and output dimensions for all dataloaders. For testing purposes.
        get_initial_guess: Function used to get the initial random guess for the gradient ascent. For testing purposes.
        ascend: Function used to do the actual ascending. For testing purposes.

    Returns:
        The MEI as a tensor and a list of model evaluations at each step of the gradient ascent process.
    """

    config = prepare_config(config, import_object)
    mei_shape = get_input_dimensions(dataloaders, get_dims)
    initial_guess = initial_image if initial_image is not None else get_initial_guess(1, *mei_shape[1:], ).cuda()
    print("shape_of_initial_guess")
    print(initial_guess.shape)
    mei, evaluations, _ = ascend(model, initial_guess, **config)
    return mei, evaluations


def prepare_config(config, import_object):
    config = prepare_optim_kwargs(config)
    return import_functions(config, import_object)


def prepare_optim_kwargs(config):
    if not config["optim_kwargs"]:
        config["optim_kwargs"] = dict()
    return config


def import_functions(config, import_object):
    for attribute in ("transform", "regularization", "gradient_f", "post_update"):
        if not config[attribute]:
            continue
        config[attribute] = import_object(config[attribute])
    return config


def get_input_dimensions(dataloaders, get_dims):
    in_name, out_name = next(iter(list(dataloaders["train"].values())[0]))._fields
    return list(get_dims(dataloaders["train"]).values())[0][in_name]
