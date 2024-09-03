import torch
import os

from ..models import se_core_full_gauss_readout

# full model keys
key = [
    {
        "model_fn": "nndichromacy.models.se_core_full_gauss_readout",
        "model_hash": "fa56487e5ead382e5ee64b0096c20043",
        "dataset_fn": "nndichromacy.datasets.static_loaders",
        "dataset_hash": "9134af609e6f41bf0ae5c0bb231b3cf4",
        "trainer_fn": "nndichromacy.training.standart_trainer",
        "trainer_hash": "0d06f037501e129d11aa288d8f22788f",
        "seed": 7000,
    },
    {
        "model_fn": "nndichromacy.models.se_core_full_gauss_readout",
        "model_hash": "fa56487e5ead382e5ee64b0096c20043",
        "dataset_fn": "nndichromacy.datasets.static_loaders",
        "dataset_hash": "9134af609e6f41bf0ae5c0bb231b3cf4",
        "trainer_fn": "nndichromacy.training.standart_trainer",
        "trainer_hash": "0d06f037501e129d11aa288d8f22788f",
        "seed": 3000,
    },
    {
        "model_fn": "nndichromacy.models.se_core_full_gauss_readout",
        "model_hash": "fa56487e5ead382e5ee64b0096c20043",
        "dataset_fn": "nndichromacy.datasets.static_loaders",
        "dataset_hash": "9134af609e6f41bf0ae5c0bb231b3cf4",
        "trainer_fn": "nndichromacy.training.standart_trainer",
        "trainer_hash": "0d06f037501e129d11aa288d8f22788f",
        "seed": 8000,
    },
    {
        "model_fn": "nndichromacy.models.se_core_full_gauss_readout",
        "model_hash": "fa56487e5ead382e5ee64b0096c20043",
        "dataset_fn": "nndichromacy.datasets.static_loaders",
        "dataset_hash": "9134af609e6f41bf0ae5c0bb231b3cf4",
        "trainer_fn": "nndichromacy.training.standart_trainer",
        "trainer_hash": "0d06f037501e129d11aa288d8f22788f",
        "seed": 5000,
    },
    {
        "model_fn": "nndichromacy.models.se_core_full_gauss_readout",
        "model_hash": "fa56487e5ead382e5ee64b0096c20043",
        "dataset_fn": "nndichromacy.datasets.static_loaders",
        "dataset_hash": "9134af609e6f41bf0ae5c0bb231b3cf4",
        "trainer_fn": "nndichromacy.training.standart_trainer",
        "trainer_hash": "0d06f037501e129d11aa288d8f22788f",
        "seed": 1000,
    },
]

model_fn = se_core_full_gauss_readout
model_config = {
    "pad_input": False,
    "stack": -1,
    "layers": 4,
    "input_kern": 9,
    "gamma_input": 6.3831,
    "gamma_readout": 0.0076,
    "hidden_dilation": 1,
    "hidden_kern": 7,
    "hidden_channels": 64,
    "n_se_blocks": 0,
    "depth_separable": True,
    "grid_mean_predictor": None,
    "share_features": False,
    "share_grid": False,
    "init_sigma": 0.1,
    "init_mu_range": 0.3,
    "gauss_type": "full",
}

data_info = data_info = {
    "26614-4-11": {
        "input_channels": 1,
        "input_dimensions": torch.Size([128, 1, 36, 64]),
        "output_dimension": 6869,
        "img_mean": 122.0,
        "img_std": 54.0,
    },
}


current_dir = os.path.dirname(__file__)
filename = os.path.join(
    current_dir, "../../data/model_weights/v1_data_driven/v1_surround_1.tar"
)
state_dict = torch.load(filename)

# load single model
v1_surround_model = model_fn(
    seed=0,
    dataloaders=None,
    **model_config,
    data_info=data_info,
)
v1_surround_model.load_state_dict(state_dict)


# load ensemble model
from mei.modules import EnsembleModel

ensemble_names = [
    "v1_surround_2.tar",
    "v1_surround_3.tar",
    "v1_surround_4.tar",
    "v1_surround_5.tar",
]

base_dir = os.path.dirname(filename)
ensemble_models = []
ensemble_models.append(v1_surround_model)

for f in ensemble_names:
    ensemble_filename = os.path.join(base_dir, f)
    ensemble_state_dict = torch.load(ensemble_filename)
    ensemble_model = model_fn(
        seed=0,
        dataloaders=None,
        **model_config,
        data_info=data_info,
    )
    ensemble_model.load_state_dict(ensemble_state_dict)
    ensemble_models.append(ensemble_model)

# Ensemble model
v1_surround_ensemble = EnsembleModel(*ensemble_models)
