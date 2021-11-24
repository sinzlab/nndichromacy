import tempfile
import os

import datajoint as dj
import torch

from nnfabrik.main import Dataset, schema
from nnfabrik.utility.dj_helpers import make_hash
from . import integration
from neuralpredictors.data.datasets import StaticImageSet


class TrainedEnsembleModelTemplate(dj.Manual):
    """TrainedEnsembleModel table template.

    To create a functional "TrainedEnsembleModel" table, create a new class that inherits from this template and
    decorate it with your preferred Datajoint schema. Next assign the trained model table of your choosing to the class
    variable called "trained_model_table". By default the created table will point to the "Dataset" table in the
    Datajoint schema called "nnfabrik.main". This behaviour can be changed by overwriting the class attribute called
    "dataset_table".
    """

    dataset_table = Dataset
    trained_model_table = None

    definition = """
    # contains ensemble ids
    -> self.dataset_table
    ensemble_hash : char(32) # the hash of the ensemble
    """

    class Member(dj.Part):
        """Member table template."""

        definition = """
        # contains assignments of trained models to a specific ensemble id
        -> master
        -> master.trained_model_table
        """

    def create_ensemble(self, key, model_keys=None):
        """Creates a new ensemble and inserts it into the table.

        Args:
            key: A dictionary representing a key that must be sufficient to restrict the dataset table to one entry. The
                models that are in the trained model table after restricting it with the provided key will be part of
                the ensemble.

        Returns:
            None.
        """
        print("creating ensemble")
        if len(self.dataset_table() & key) != 1:
            raise ValueError(
                "Provided key not sufficient to restrict dataset table to one entry!"
            )
        dataset_key = (self.dataset_table().proj() & key).fetch1()
        if model_keys is None:
            models = (self.trained_model_table().proj() & key).fetch(as_dict=True)
        else:
            print(
                "model dictionties were passed - creating ensemble with {} models".format(
                    len(model_keys)
                )
            )
            models = model_keys
        ensemble_table_key = dict(
            dataset_key, ensemble_hash=integration.hash_list_of_dictionaries(models)
        )
        self.insert1(ensemble_table_key)

        self.Member().insert([{**ensemble_table_key, **m} for m in models])

    def load_model(self, key=None):
        """Wrapper to preserve the interface of the trained model table."""
        return integration.load_ensemble_model(
            self.Member, self.trained_model_table, key=key
        )


class CSRFV1SelectorTemplate(dj.Computed):
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

    _key_source = Dataset & dict(dataset_fn="csrf_v1")

    def make(self, key):
        dataset_config = (Dataset & key).fetch1("dataset_config")
        mappings = integration.get_mappings(dataset_config, key)
        self.insert(mappings)

    def get_output_selected_model(self, model, key):
        neuron_pos, session_id = (self & key).fetch1("neuron_position", "session_id")
        return integration.get_output_selected_model(neuron_pos, session_id, model)


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

    _key_source = Dataset & dict(dataset_fn="mouse_static_loaders")

    def make(self, key):
        dataset_config = (Dataset & key).fetch1("dataset_config")

        path = dataset_config["paths"][0]
        dat = StaticImageSet(path, "images", "responses")
        neuron_ids = dat.neurons.unit_ids

        data_key = path.split("static")[-1].split(".")[0].replace("preproc", "")

        mappings = []
        for neuron_pos, neuron_id in enumerate(neuron_ids):
            mappings.append(
                dict(
                    key,
                    neuron_id=neuron_id,
                    neuron_position=neuron_pos,
                    session_id=data_key,
                )
            )

        self.insert(mappings)

    def get_output_selected_model(self, model, key):
        neuron_pos, session_id = (self & key).fetch1("neuron_position", "session_id")
        return integration.get_output_selected_model(neuron_pos, session_id, model)


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
        self.insert1(
            dict(
                method_fn=method_fn,
                method_hash=make_hash(method_config),
                method_config=method_config,
            )
        )

    def generate_mei(self, dataloader, model, key):
        method_fn, method_config = (self & key).fetch1("method_fn", "method_config")
        method_fn = integration.import_module(method_fn)
        mei, evaluations = method_fn(dataloader, model, method_config)
        return dict(key, evaluations=evaluations, mei=mei)


class MEITemplate(dj.Computed):
    """MEI table template.

    To create a functional "MEI" table, create a new class that inherits from this template and decorate it with your
    preferred Datajoint schema. Next assign your trained model (or trained ensemble model) and your selector table to
    the class variables called "trained_model_table" and "selector_table". By default, the created table will point to
    the "MEIMethod" table in the Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting
    the class attribute called "method_table".
    """

    method_table = MEIMethod
    trained_model_table = None
    selector_table = None

    definition = """
    # contains maximally exciting images (MEIs)
    -> self.method_table
    -> self.trained_model_table
    -> self.selector_table
    ---
    mei                 : attach@minio  # the MEI as a tensor
    evaluations         : longblob      # list of function evaluations at each iteration in the mei generation process 
    """

    def __init__(self, cache_size_limit=10):
        """Initializes MEITemplate.

        Args:
            cache_size_limit: An integer indicating the maximum number of cached models.
        """
        super().__init__()
        self.model_loader = integration.ModelLoader(
            self.trained_model_table, cache_size_limit=cache_size_limit
        )

    def make(self, key):
        dataloaders, model = self.model_loader.load(key=key)
        output_selected_model = self.selector_table().get_output_selected_model(
            model, key
        )
        mei_entity = self.method_table().generate_mei(
            dataloaders, output_selected_model, key
        )
        self._insert_mei(mei_entity)

    def _insert_mei(self, mei_entity):
        """Saves the MEI to a temporary directory and inserts the prepared entity into the table."""
        mei = mei_entity.pop("mei").squeeze()
        filename = make_hash(mei_entity) + ".pth.tar"
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, filename)
            torch.save(mei, filepath)
            mei_entity["mei"] = filepath
            self.insert1(mei_entity)
