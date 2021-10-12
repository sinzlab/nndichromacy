import torch
import numpy as np
from insilico_stimuli.stimuli import GaborSet
from insilico_stimuli.tables.main import ExperimentMethod
from tqdm import tqdm

# Images returned by GaborSet of inSilico.stimuli are returned in this order:
param_order_general = [
    "locations",
    "sizes",
    "spatial_frequencies",
    "orientations",
    "phases",
]


def isoresponse_gabors(
    StimulusSet,
    model=None,
    previous_experiment=None,
    unit=None,
    seed=None,
    batch_size=1000,
    config_dict=None,
):
    """
    Args:
        StimulusSet:
        model:
        previous_experiment:
        unit:
        seed:
        config: dict,

    Returns:
    """

    img_size = StimulusSet.canvas_size
    previous_method_config = (ExperimentMethod & previous_experiment).fetch1(
        "method_config"
    )
    norm = previous_method_config["fixed_norm"]

    # get optimal parameters from the gradient descent optimizer
    output = previous_experiment["output"]
    optimal_lambda = 1 / output["Lambda"].item()
    optimal_theta = -1 * output["theta"].item()
    optimal_psi = output["psi"].item()
    optimal_sigma = output["sigma"].item() * 4
    optimal_location = [
        img_size[0] / 2 + output["center"][0],
        img_size[1] / 2 + output["center"][1],
    ]

    gabor_params = dict()

    if "sizes" not in config_dict:
        gabor_params["sizes"] = [optimal_sigma]
    else:
        start = config_dict["sizes"].get("start", 1)
        end = config_dict["sizes"].get("end", 100)
        n_stimuli = config_dict["sizes"].get("n_stimuli", 100)
        gabor_params["sizes"] = [
            *(np.linspace(start, optimal_sigma, n_stimuli // 2, endpoint=False)),
            *(np.linspace(optimal_sigma, end, n_stimuli // 2 + 1)),
        ]

    if "orientations" not in config_dict:
        gabor_params["orientations"] = [optimal_theta]
    else:
        start = config_dict["orientations"].get("start", optimal_theta - np.pi / 2)
        end = config_dict["orientations"].get("end", optimal_theta + np.pi / 2)
        n_stimuli = config_dict["orientations"].get("n_stimuli", 100)
        gabor_params["orientations"] = [
            *(np.linspace(start, optimal_theta, n_stimuli // 2, endpoint=False)),
            *(np.linspace(optimal_theta, end, n_stimuli // 2 + 1)),
        ]

    if "phases" not in config_dict:
        gabor_params["phases"] = [optimal_psi]
    else:
        start = config_dict["phases"].get("start", optimal_psi - np.pi)
        end = config_dict["phases"].get("end", optimal_psi + np.pi)
        n_stimuli = config_dict["phases"].get("n_stimuli", 100)
        gabor_params["phases"] = [
            *(np.linspace(start, optimal_psi, n_stimuli // 2, endpoint=False)),
            *(np.linspace(optimal_psi, end, n_stimuli // 2 + 1)),
        ]

    if "spatial_frequencies" not in config_dict:
        gabor_params["spatial_frequencies"] = [optimal_lambda]
    else:
        start = config_dict["spatial_frequencies"].get("start", 0.01)
        end = config_dict["spatial_frequencies"].get("end", 0.25)
        n_stimuli = config_dict["spatial_frequencies"].get("n_stimuli", 100)
        gabor_params["spatial_frequencies"] = [
            *(np.linspace(start, optimal_lambda, n_stimuli // 2, endpoint=False)),
            *(np.linspace(optimal_lambda, end, n_stimuli // 2 + 1)),
        ]

    if "locations" not in config_dict:
        gabor_params["locations"] = [optimal_location]
    else:
        sample_range = config_dict["locations"].get("sample_range", 15)
        start = config_dict["locations"].get(
            "start",
            [optimal_location[0] - sample_range, optimal_location[1] - sample_range],
        )
        end = config_dict["locations"].get(
            "end",
            [optimal_location[0] + sample_range, optimal_location[1] + sample_range],
        )
        n_stimuli = config_dict["locations"].get("n_stimuli", 100)

        locations_inserted = 0
        locations = []
        for width in np.linspace(start[0], end[0], int(np.sqrt(n_stimuli))):
            for height in np.linspace(start[1], end[1], int(np.sqrt(n_stimuli))):
                locations.append([width, height])
                locations_inserted += 1
                if locations_inserted == n_stimuli // 2:
                    locations.append(optimal_location)
        gabor_params["locations"] = locations

    # Gabor stimulus - Fixed Inputs
    canvas_size = img_size
    contrasts = [1.0]
    grey_levels = [0.0]
    eccentricities = [0.0]

    # Instantiate the Gabor class
    print("Gabor with Dict")
    full_gabor_config = dict(
        canvas_size=canvas_size,
        sizes=gabor_params["sizes"],
        spatial_frequencies=gabor_params["spatial_frequencies"],
        contrasts=contrasts,
        orientations=gabor_params["orientations"],
        phases=gabor_params["phases"],
        grey_levels=grey_levels,
        eccentricities=eccentricities,
        locations=gabor_params["locations"],
        relative_sf=False,
    )
    gabor_set = GaborSet(**full_gabor_config)

    activations = []
    batch_size = (
        batch_size
        if np.prod(gabor_set.num_params()) > batch_size
        else np.prod(gabor_set.num_params())
    )
    print(f"Total Stimuli: {np.prod(gabor_set.num_params())}")
    print(f"Batch size: {batch_size}")
    print(f"Total Iterations: {np.ceil(np.prod(gabor_set.num_params()) / batch_size)}")
    for b in tqdm(gabor_set.image_batches(batch_size)):
        x = torch.from_numpy(b)
        x_norm = torch.norm(x.view(len(x), -1), dim=-1)
        x = x * (norm / x_norm).view(len(x), *[1] * (x.dim() - 1))
        with torch.no_grad():
            activations.append(
                model(x[:, None, ...].to(torch.float32).cuda()).cpu().numpy()[:, unit]
            )
    activations = np.hstack(activations)

    param_order, activation_reshape_list = [], []
    output = dict()

    for key in param_order_general:
        if key in config_dict:
            param_order.append(key)
            output[key] = gabor_params[key]
            activation_reshape_list.append(len(gabor_params[key]))

    output["activations"] = activations.reshape(*activation_reshape_list)
    output["param_order"] = param_order
    output["full_gabor_config"] = full_gabor_config
    score = activations.max()

    return output, score
