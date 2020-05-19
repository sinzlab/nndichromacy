from __future__ import annotations

import datajoint as dj
from nnfabrik.main import Dataset
from .from_nnfabrik import TrainedModel
from mlutils.data.datasets import StaticImageSet, FileTreeDataset
from featurevis import integration
from featurevis import mixins
from featurevis.main import MEITemplate, MEISeed
from torch.nn import Module
from torch.utils.data import DataLoader

from featurevis.integration import ConstrainedOutputModel
from nnfabrik.utility.dj_helpers import CustomSchema

from typing import Callable, Iterable, Mapping, Optional, Tuple, Dict, Any
Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]


schema = CustomSchema(dj.config.get('schema_name', 'nnfabrik_core'))


class MouseSelectorTemplate(dj.Computed):

    dataset_table = Dataset
    definition = """
    # contains information that can be used to map a neuron's id to its corresponding integer position in the output of
    # the model. 
    -> self.dataset_table
    unit_id       : int               # unique neuron identifier
    data_key        : varchar(255)      # unique session identifier
    ---
    unit_index : int                    # integer position of the neuron in the model's output 
    """

    constrained_output_model = ConstrainedOutputModel

    def make(self, key):
        dataset_config = (Dataset & key).fetch1("dataset_config")

        path = dataset_config["paths"][0]
        file_tree = dataset_config.get("file_tree", False)
        dat = StaticImageSet(path, 'images', 'responses') if not file_tree else FileTreeDataset(path, 'images', 'responses')
        neuron_ids = dat.neurons.unit_ids

        data_key = path.split('static')[-1].split('.')[0].replace('preproc', '').replace('_nobehavior','')

        mappings = []
        for neuron_pos, neuron_id in enumerate(neuron_ids):
            mappings.append(dict(key, unit_id=neuron_id, unit_index=neuron_pos, data_key=data_key))

        self.insert(mappings)

    def get_output_selected_model(self, model: Module, key: Key) -> constrained_output_model:
        unit_index, data_key = (self & key).fetch1("unit_index", "data_key")
        return self.constrained_output_model(model, unit_index, forward_kwargs=dict(data_key=data_key))


@schema
class MEISelector(MouseSelectorTemplate):
    dataset_table = Dataset


@schema
class MEIMethod(mixins.MEIMethodMixin, dj.Lookup):
    seed_table = MEISeed


@schema
class TrainedEnsembleModel(mixins.TrainedEnsembleModelTemplateMixin, dj.Manual):
    dataset_table = Dataset
    trained_model_table = TrainedModel
    class Member(mixins.TrainedEnsembleModelTemplateMixin.Member, dj.Part):
        """Member table template."""
        pass


@schema
class MEI(mixins.MEITemplateMixin, dj.Computed):
    """MEI table template.

    To create a functional "MEI" table, create a new class that inherits from this template and decorate it with your
    preferred Datajoint schema. Next assign your trained model (or trained ensemble model) and your selector table to
    the class variables called "trained_model_table" and "selector_table". By default, the created table will point to
    the "MEIMethod" table in the Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting
    the class attribute called "method_table".
    """

    trained_model_table = TrainedEnsembleModel
    selector_table = MEISelector
    method_table = MEIMethod
    seed_table = MEISeed