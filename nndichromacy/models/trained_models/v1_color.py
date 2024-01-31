import torch
import os

from nnfabrik.builder import get_model

# full model keys
key = [{'model_fn': 'nndichromacy.models.se_core_full_gauss_readout',
  'model_hash': 'af90cf960a14d3ea4b8d6138d4510693',
  'dataset_fn': 'nndichromacy.datasets.static_loaders',
  'dataset_hash': 'a8391e6563fb1dc239215e01119f1bd5',
  'trainer_fn': 'nndichromacy.training.standart_trainer',
  'trainer_hash': '0d06f037501e129d11aa288d8f22788f',
  'seed': 6000},
 {'model_fn': 'nndichromacy.models.se_core_full_gauss_readout',
  'model_hash': 'af90cf960a14d3ea4b8d6138d4510693',
  'dataset_fn': 'nndichromacy.datasets.static_loaders',
  'dataset_hash': 'a8391e6563fb1dc239215e01119f1bd5',
  'trainer_fn': 'nndichromacy.training.standart_trainer',
  'trainer_hash': '0d06f037501e129d11aa288d8f22788f',
  'seed': 8000},
 {'model_fn': 'nndichromacy.models.se_core_full_gauss_readout',
  'model_hash': 'af90cf960a14d3ea4b8d6138d4510693',
  'dataset_fn': 'nndichromacy.datasets.static_loaders',
  'dataset_hash': 'a8391e6563fb1dc239215e01119f1bd5',
  'trainer_fn': 'nndichromacy.training.standart_trainer',
  'trainer_hash': '0d06f037501e129d11aa288d8f22788f',
  'seed': 9000},
 {'model_fn': 'nndichromacy.models.se_core_full_gauss_readout',
  'model_hash': 'af90cf960a14d3ea4b8d6138d4510693',
  'dataset_fn': 'nndichromacy.datasets.static_loaders',
  'dataset_hash': 'a8391e6563fb1dc239215e01119f1bd5',
  'trainer_fn': 'nndichromacy.training.standart_trainer',
  'trainer_hash': '0d06f037501e129d11aa288d8f22788f',
  'seed': 2000},
 {'model_fn': 'nndichromacy.models.se_core_full_gauss_readout',
  'model_hash': 'af90cf960a14d3ea4b8d6138d4510693',
  'dataset_fn': 'nndichromacy.datasets.static_loaders',
  'dataset_hash': 'a8391e6563fb1dc239215e01119f1bd5',
  'trainer_fn': 'nndichromacy.training.standart_trainer',
  'trainer_hash': '0d06f037501e129d11aa288d8f22788f',
  'seed': 3000}]

model_fn = 'nndichromacy.models.se_core_full_gauss_readout'
model_config = {'pad_input': False,
  'stack': -1,
  'layers': 4,
  'input_kern': 9,
  'gamma_input': 6.3831,
  'gamma_readout': 0.0076,
  'hidden_dilation': 1,
  'hidden_kern': 7,
  'hidden_channels': 64,
  'n_se_blocks': 0,
  'depth_separable': True,
  'grid_mean_predictor':None,
  'share_features': False,
  'share_grid': False,
  'init_sigma': 0.1,
  'init_mu_range': 0.3,
  'gauss_type': 'full'}

data_info = {'25311-4-6': {'input_channels': 2,
  'input_dimensions': torch.Size([128, 2, 36, 64]),
  'output_dimension': 478,
  'img_mean': 122.0,
  'img_std': 54.0},
 '25311-7-34': {'input_channels': 2,
  'input_dimensions': torch.Size([128, 2, 36, 64]),
  'output_dimension': 623,
  'img_mean': 122.0,
  'img_std': 54.0},
 '25311-10-26': {'input_channels': 2,
  'input_dimensions': torch.Size([128, 2, 36, 64]),
  'output_dimension': 658,
  'img_mean': 122.0,
  'img_std': 54.0}}


current_dir = os.path.dirname(__file__)
filename = os.path.join(current_dir, '../../data/model_weights/v1_data_driven/v1_color_1.tar')
state_dict = torch.load(filename)

# load single model
v1_color_model = get_model(
    model_fn, model_config, seed=10, data_info=data_info, state_dict=state_dict
)


# load ensemble model
from mei.modules import EnsembleModel

ensemble_names = ['v1_color_2.tar',
 'v1_color_3.tar',
 'v1_color_4.tar',
 'v1_color_5.tar',]

base_dir = os.path.dirname(filename)
ensemble_models = []
ensemble_models.append(v1_data_driven_sota)

for f in ensemble_names:
    ensemble_filename = os.path.join(base_dir, f)
    ensemble_state_dict = torch.load(ensemble_filename)
    ensemble_model = get_model(
        model_fn, model_config, seed=10, data_info=data_info, state_dict=ensemble_state_dict
    )
    ensemble_models.append(ensemble_model)

# Ensemble model
v1_color_ensemble = EnsembleModel(*ensemble_models)

