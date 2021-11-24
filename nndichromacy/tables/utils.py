from __future__ import annotations
from typing import Dict, Any, List, Callable

import numpy as np
import torch
from dataport.bcm.color_mei.utils import rescale_frame
from nnfabrik.utility.nn_helpers import get_dims_for_loader_dict
import torchvision

from mei.methods import get_input_dimensions, import_object
from mei.methods import get_dims_for_loader_dict as get_dims


def get_image_data_from_dataset(
    dat,
    image_class,
    image_id,
    return_behavior=False,
    image_repeat=None,
    return_image=None,
):
    if "image_id" in dir(dat.trial_info):
        image_ids = dat.trial_info.image_id
        image_classes = dat.trial_info.image_class
    elif "colorframeprojector_image_id" in dir(dat.trial_info):
        image_ids = dat.trial_info.colorframeprojector_image_id
        image_classes = dat.trial_info.colorframeprojector_image_class
    elif "frame_image_id" in dir(dat.trial_info):
        image_ids = dat.trial_info.frame_image_id
        image_classes = dat.trial_info.frame_image_class

    trial_idx = np.where((image_ids == image_id) & (image_classes == image_class))[0]
    if (len(trial_idx) > 1) and (image_repeat is None):
        raise ValueError(
            "More than 1 repetition for the probe img present. The kwarg image_repeat has to be set in the method config"
        )
    if image_repeat is not None:
        trial_idx = trial_idx[image_repeat]
    print("Repeat Used: ", image_repeat)
    responses = dat[trial_idx].responses
    print(responses.shape)

    behavior = dat[trial_idx].behavior if "behavior" in dat.data_keys else None
    print(behavior)
    pupil_center = (
        dat[trial_idx].pupil_center if "pupil_center" in dat.data_keys else None
    )
    if return_image is True:
        return dat[trial_idx].images
    else:
        return responses if return_behavior is False else (behavior, pupil_center)


def extend_img_with_behavior(img, behavior):
    return torch.cat(
        [img, *[torch.ones(1, 1, *img.shape[2:]).to(img.device) * i for i in behavior]],
        dim=1,
    )


def preprocess_img_for_reconstruction(
    img, img_size, img_statistics, dataloaders, device="cuda"
):
    """
    Turn the initial image from a numpy array (height x width x n_channels)
    into a torch.Tensor on the specified device.
    Args:
        img: np.array (height, width, n_channels)
        img_size (tuple): desired (height, width) of the img, has to match the models expectation
        img_statistics (tuple): (mean, std)
        device: "cuda" or "cpu"

    Returns: torch.Tensor

    """
    if len(img.shape) < 4:
        if img.shape[2] not in [1, 2, 3]:
            raise ValueError("unexpected image shape")
        if not isinstance(img, np.ndarray):
            img.detach().cpu().numpy()
        if img.shape[:2] != img_size:
            img = rescale_frame(img, img_size=img_size)
        img = torch.from_numpy(img).permute(2, 0, 1)[None, ...].to(device)

    if img.shape[0] > 1:
        raise ValueError("Batch dimension of the reconstructed img has to be 1.")

    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).to(device)

    data_key = list((dataloaders["train"].keys()))[0]
    dimensions_images = get_dims_for_loader_dict(dataloaders["train"])[data_key].get(
        "images", 0
    )
    dimensions_behavior = get_dims_for_loader_dict(dataloaders["train"])[data_key].get(
        "behavior", 0
    )
    if (dimensions_images[1] - dimensions_behavior[1]) > 1:
        img = img[:, 1:, ...]
    else:
        img = torchvision.transforms.Grayscale(num_output_channels=1)(img)

    img = (img - img_statistics[0]).to(device) / img_statistics[1]
    return img


def get_behavior_from_method_config(method_config) -> tuple:
    """
    Parases the method config for the settings of behavior as image channels

    Args:
        method_config (dict): A method config from the MEIMethod or ReconstructionMethod table.

    Returns (tuple): behavior
    """
    initial = method_config.get("initial")
    if "selected_channels" not in initial["kwargs"]:
        pupil = initial["kwargs"].get("channel_0", None)
        dpupil = initial["kwargs"].get("channel_1", None)
        running = initial["kwargs"].get("channel_2", None)
    else:
        pupil = initial["kwargs"]["selected_values"][0]
        dpupil = initial["kwargs"]["selected_values"][1]
        running = initial["kwargs"]["selected_values"][2]
    behavior = (pupil, dpupil, running)
    kwargs = method_config.get("model_forward_kwargs", dict())
    return behavior, kwargs


def get_initial_image(
    dataloaders,
    data_key,
    method_config,
):
    shape = method_config.get(
        "mei_shape", get_input_dimensions(dataloaders, get_dims, data_key=data_key)
    )

    create_initial_guess = import_object(
        method_config["initial"]["path"], method_config["initial"]["kwargs"]
    )
    initial_guess = create_initial_guess(1, *shape[1:]).to(method_config["device"])
    return initial_guess


def process_image(initial_img, image):
    img_shape = image.shape
    initial_img[:, : img_shape[1], ...] = image

    return initial_img
