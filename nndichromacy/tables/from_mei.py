import datajoint as dj
from featurevis.main import TrainedEnsembleModelTemplate, CSRFV1SelectorTemplate, MEIMethod, MEITemplate
from nnfabrik.main import Dataset
from .from_nnfabrik import TrainedModel
from mlutils.data.datasets import StaticImageSet, FileTreeDataset
from featurevis import integration
from ..mei.helpers import get_neuron_mappings, get_real_mappings
from ..mei.regularizers import rgb_initial_guess
from nnfabrik.utility.dj_helpers import make_hash
from featurevis import integration
from featurevis.methods import gradient_ascent

import torch

schema = dj.schema(dj.config.get('schema_name', 'nnfabrik_core'))

class MouseSelectorTemplate(dj.Computed):
    """CSRF V1 selector table template.

    To create a functional "CSRFV1Selector" table, create a new class that inherits from this template and decorate it
    with your preferred Datajoint schema. By default, the created table will point to the "Dataset" table in the
    Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting the class attribute called
    "dataset_table".
    """

    dataset_table = Dataset

    definition = """
    # contains information that can be used to map a neuron's id to its corresponding integer position in the output of
    # the model. 
    -> self.dataset_table
    neuron_id       : smallint unsigned # unique neuron identifier
    ---
    neuron_position : smallint unsigned # integer position of the neuron in the model's output 
    session_id      : varchar(13)       # unique session identifier
    """

    #_key_source = Dataset & dict(dataset_fn="mouse_static_loaders")

    def make(self, key):
        dataset_config = (Dataset & key).fetch1("dataset_config")

        path = dataset_config["paths"][0]
        file_tree = dataset_config.get("file_tree", False)
        dat = StaticImageSet(path, 'images', 'responses') if not file_tree else FileTreeDataset(path, 'images', 'responses')
        neuron_ids = dat.neurons.unit_ids

        data_key = path.split('static')[-1].split('.')[0].replace('preproc', '').replace('_nobehavior','')

        mappings = []
        for neuron_pos, neuron_id in enumerate(neuron_ids):
            mappings.append(dict(key, neuron_id=neuron_id, neuron_position=neuron_pos, session_id=data_key))

        self.insert(mappings)

    def get_output_selected_model(self, model, key):
        neuron_pos, session_id = (self & key).fetch1("neuron_position", "session_id")
        return integration.get_output_selected_model(neuron_pos, session_id, model)


@schema
class TrainedEnsembleModel(TrainedEnsembleModelTemplate):
    dataset_table = Dataset
    trained_model_table = TrainedModel

    def load_model(self, key=None, include_dataloader=True, include_state_dict=True):
        """Wrapper to preserve the interface of the trained model table."""
        return integration.load_ensemble_model(self.Member,
                                               self.trained_model_table,
                                               key=key,
                                               include_dataloader=include_dataloader,
                                               include_state_dict=include_state_dict)


@schema
class MouseSelector(MouseSelectorTemplate):
    dataset_table = Dataset


@schema
class MEIMethod(dj.Lookup):
    definition = """
    # contains methods for generating MEIs and their configurations.
    method_fn                           : varchar(64)   # name of the method function
    method_hash                         : varchar(32)   # hash of the method config
    ---
    method_config                       : longblob      # method configuration object
    method_ts       = CURRENT_TIMESTAMP : timestamp     # UTZ timestamp at time of insertion
    """

    def add_method(self, method_fn, method_config):
        self.insert1(dict(method_fn=method_fn, method_hash=make_hash(method_config), method_config=method_config))

    def generate_mei(self, dataloader, model, key):
        method_fn, method_config = (self & key).fetch1("method_fn", "method_config")
        method_fn = integration.import_module(method_fn)
        if 'get_initial_guess' in method_config:
            get_initial_guess = integration.import_module(method_config.pop("get_initial_guess"))
        else:
            get_initial_guess = torch.randn

        mei, evaluations = method_fn(dataloader, model, method_config, get_initial_guess=get_initial_guess)
        return dict(key, evaluations=evaluations, mei=mei)


@schema
class MEI(MEITemplate):
    method_table = MEIMethod
    trained_model_table = TrainedEnsembleModel
    selector_table = MouseSelector

