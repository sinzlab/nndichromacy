from __future__ import annotations
from typing import Dict, Any, List, Callable

import torch
from torch.nn import Module, ModuleList
from torch.utils.data import DataLoader

Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]

import warnings
from functools import partial

import datajoint as dj
from nnfabrik.main import Dataset
from nnfabrik.utility.dj_helpers import CustomSchema, cleanup_numpy_scalar, make_hash
from nnfabrik.builder import resolve_fn, get_data
from mei import mixins
from mei.main import MEITemplate, MEISeed
from mei.modules import ConstrainedOutputModel
from dataport.bcm.color_mei.schema import StaticImage
from .from_nnfabrik import TrainedModel
from .utils import (
    get_image_data_from_dataset,
    extend_img_with_behavior,
    get_behavior_from_method_config,
    preprocess_img_for_reconstruction,
)

schema = CustomSchema(dj.config.get("nnfabrik.schema_name", "nnfabrik_core"))
resolve_target_fn = partial(resolve_fn, default_base="targets")


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
        dataloaders = (Dataset & key).get_dataloader()
        data_keys = list(dataloaders["train"].keys())

        mappings = []
        for data_key in data_keys:
            dat = dataloaders["train"][data_key].dataset
            try:
                neuron_ids = dat.neurons.unit_ids
            except AttributeError:
                warnings.warn(
                    "unit_ids were not found in the dataset - using indices 0-N instead"
                )
                neuron_ids = range(dat.responses.shape[1])
            for neuron_pos, neuron_id in enumerate(neuron_ids):
                mappings.append(
                    dict(
                        key, unit_id=neuron_id, unit_index=neuron_pos, data_key=data_key
                    )
                )

        self.insert(mappings)

    def get_output_selected_model(
        self, model: Module, key: Key
    ) -> constrained_output_model:
        unit_index, data_key = (self & key).fetch1("unit_index", "data_key")
        return self.constrained_output_model(
            model, unit_index, forward_kwargs=dict(data_key=data_key)
        )


@schema
class MEISelector(MouseSelectorTemplate):
    dataset_table = Dataset


@schema
class MEIMethod(mixins.MEIMethodMixin, dj.Lookup):
    seed_table = MEISeed
    optional_names = optional_names = (
        "transform",
        "regularization",
        "precondition",
        "postprocessing",
    )

    def generate_mei(
        self, dataloaders: Dataloaders, model: Module, key: Key, seed: int
    ) -> Dict[str, Any]:
        method_fn, method_config = (self & key).fetch1("method_fn", "method_config")
        self.insert_key_in_ops(method_config=method_config, key=key)
        method_fn = self.import_func(method_fn)
        mei, score, output = method_fn(dataloaders, model, method_config, seed)
        return dict(key, mei=mei, score=score, output=output)

    def insert_key_in_ops(self, method_config, key):
        for k, v in method_config.items():
            if k in self.optional_names:
                if "key" in v["kwargs"]:
                    v["kwargs"]["key"] = key


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


@schema
class MEIScore(dj.Computed):
    """
    A template for a MEI scoring table.
    """

    mei_table = MEI
    measure_attribute = "score"
    function_kwargs = {}
    external_download_path = None

    # table level comment
    table_comment = "A template table for storing results/scores of a MEI"

    @property
    def definition(self):
        definition = """
                    # {table_comment}
                    -> self.mei_table
                    ---
                    {measure_attribute}:      float     # A template for a computed score of a trained model
                    {measure_attribute}_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
                    """.format(
            table_comment=self.table_comment, measure_attribute=self.measure_attribute
        )
        return definition

    @staticmethod
    def measure_function(mei, **kwargs):
        raise NotImplementedError("Scoring Function has to be implemented")

    def get_mei(self, key):
        mei = torch.load(
            (self.mei_table & key).fetch1(
                "mei", download_path=self.external_download_path
            )
        )
        return mei

    def make(self, key):
        mei = self.get_mei(key=key)
        score = self.measure_function(mei, **self.function_kwargs)
        key[self.measure_attribute] = score
        self.insert1(key, ignore_extra_fields=True)


@schema
class ReconMethod(mixins.MEIMethodMixin, dj.Lookup):
    seed_table = MEISeed
    optional_names = optional_names = (
        "transform",
        "regularization",
        "precondition",
        "postprocessing",
    )

    def generate_mei(
        self, dataloaders: Dataloaders, model: Module, key: Key, seed: int
    ) -> Dict[str, Any]:
        method_fn, method_config = (self & key).fetch1("method_fn", "method_config")
        self.insert_key_in_ops(method_config=method_config, key=key)
        method_fn = self.import_func(method_fn)
        mei, score, output = method_fn(dataloaders, model, method_config, seed)
        return dict(key, mei=mei, score=score, output=output)

    def insert_key_in_ops(self, method_config, key):
        for k, v in method_config.items():
            if k in self.optional_names:
                if "key" in v["kwargs"]:
                    v["kwargs"]["key"] = key


@schema
class ReconTargetFunction(dj.Manual):
    definition = """
    target_fn:       varchar(64)
    target_hash:     varchar(64)
    ---
    target_config:   longblob
    target_comment:  varchar(128)
    """

    resolve_fn = resolve_target_fn

    @property
    def fn_config(self):
        target_fn, target_config = self.fetch1("target_fn", "target_config")
        target_config = cleanup_numpy_scalar(target_config)
        return target_fn, target_config

    def add_entry(
        self, target_fn, target_config, target_comment="", skip_duplicates=False
    ):
        """
        Add a new entry to the TargetFunction table.

        Args:
            target_fn (string) - name of a callable object. If name contains multiple parts separated by `.`, this is assumed to be found in a another module and
                dynamic name resolution will be attempted. Other wise, the name will be checked inside `targets` subpackage.
            target_config (dict) - Python dictionary containing keyword arguments for the target_fn
            dataset_comment - Optional comment for the entry.
            target_comment - If True, no error is thrown when a duplicate entry (i.e. entry with same target_fn and target_config) is found.

        Returns:
            key - key in the table corresponding to the new (or possibly existing, if skip_duplicates=True) entry.
        """

        try:
            resolve_target_fn(target_fn)
        except (NameError, TypeError) as e:
            warnings.warn(str(e) + "\nTable entry rejected")
            return

        target_hash = make_hash(target_config)
        key = dict(
            target_fn=target_fn,
            target_hash=target_hash,
            target_config=target_config,
            target_comment=target_comment,
        )

        existing = self.proj() & key
        if existing:
            if skip_duplicates:
                warnings.warn("Corresponding entry found. Skipping...")
                key = (self & (existing)).fetch1()
            else:
                raise ValueError("Corresponding entry already exists")
        else:
            self.insert1(key)

        return key

    def get_target_fn(self, key=None, **kwargs):
        if key is None:
            key = self.fetch("KEY")
        target_fn, target_config = (self & key).fn_config
        return partial(self.resolve_fn(target_fn), **target_config, **kwargs)


@schema
class ReconTargetUnit(dj.Manual):
    definition = """
    -> Dataset
    unit_hash:                      varchar(64)    # hash the list of unit IDs and data_key
    data_key:                       varchar(64)    # 

    ---
    unit_ids:                       longblob       # list of unit_ids 
    unit_comment:                   varchar(128)
    """
    dataset_table = Dataset

    def add_entry(
        self,
        dataset_fn,
        dataset_hash,
        unit_ids=None,
        data_key=None,
        unit_comment="",
        skip_duplicates=False,
    ):
        """
        Add a new entry to the TargetFunction table.

        Args:
            target_fn (string) - name of a callable object. If name contains multiple parts separated by `.`, this is assumed to be found in a another module and
                dynamic name resolution will be attempted. Other wise, the name will be checked inside `targets` subpackage.
            target_config (dict) - Python dictionary containing keyword arguments for the target_fn
            dataset_comment - Optional comment for the entry.
            target_comment - If True, no error is thrown when a duplicate entry (i.e. entry with same target_fn and target_config) is found.

        Returns:
            key - key in the table corresponding to the new (or possibly existing, if skip_duplicates=True) entry.
        """

        dataloaders = (
            self.dataset_table & dict(dataset_fn=dataset_fn, dataset_hash=dataset_hash)
        ).get_dataloader()
        data_keys = list(dataloaders["train"].keys())
        if data_key is None and len(data_keys) == 1:
            data_key = data_keys[0]
        elif data_key is None and len(data_keys) > 1:
            raise ValueError(
                "Multiple data_keys found. Data_key that is used for optimization has to be specified."
            )

        unit_ids = [] if unit_ids is None else unit_ids

        unit_hash = make_hash([unit_ids, data_key])
        key = dict(
            dataset_fn=dataset_fn,
            dataset_hash=dataset_hash,
            unit_hash=unit_hash,
            unit_ids=unit_ids,
            data_key=data_key,
            unit_comment=unit_comment,
        )
        existing = self.proj() & key
        if existing:
            if skip_duplicates:
                warnings.warn("Corresponding entry found. Skipping...")
                key = (self & (existing)).fetch1()
            else:
                raise ValueError("Corresponding entry already exists")
        else:
            self.insert1(key)

        return key


@schema
class ReconType(dj.Lookup):
    definition = """
    # specifies wether the reconstruction is based on the models' or the real neurons' responses.
    recon_type : varchar(64)
    """
    contents = [["neurons"], ["model"]]


@schema
class ReconObjective(dj.Computed):
    target_fn_table = ReconTargetFunction
    target_unit_table = ReconTargetUnit
    constrained_output_model = ConstrainedOutputModel

    @property
    def definition(self):
        definition = """
        -> self.target_fn_table 
        -> self.target_unit_table
        ---
        objective_comment:  varchar(128)
        """
        return definition

    def make(self, key):
        comments = []
        comments.append((self.target_fn_table & key).fetch1("target_comment"))
        comments.append((self.target_unit_table & key).fetch1("unit_comment"))

        key["objective_comment"] = ", ".join(comments)
        self.insert1(key)

    def get_output_selected_model(
        self,
        model: Module,
        target_fn: Callable,
        unit_ids: List,
        data_key: str,
    ) -> constrained_output_model:

        return self.constrained_output_model(
            model, unit_ids, target_fn, forward_kwargs=dict(data_key=data_key)
        )


@schema
class Reconstruction(mixins.MEITemplateMixin, dj.Computed):
    definition = """
    # contains maximally exciting images (MEIs)
    -> self.method_table
    -> self.trained_model_table
    -> self.selector_table
    -> StaticImage.Image
    -> ReconType
    -> self.seed_table
    ---
    mei                 : attach@minio  # the MEI as a tensor
    score               : float         # some score depending on the used method function
    output              : attach@minio  # object returned by the method function
    """

    trained_model_table = TrainedEnsembleModel
    selector_table = ReconObjective
    target_fn_table = ReconTargetFunction
    target_unit_table = ReconTargetUnit
    method_table = ReconMethod
    recon_type_table = ReconType
    base_image_table = StaticImage.Image
    seed_table = MEISeed
    reconstruction_size = (36, 64)

    def get_model_responses(self, model, key, image, behavior=None, **kwargs):
        model.eval()
        model.cuda()
        if behavior is not None:
            image = extend_img_with_behavior(image, behavior)
        with torch.no_grad():
            responses = model(
                image,
                data_key=key["data_key"],
                **kwargs,
            )
        return responses

    def get_neuronal_responses(self, dataloaders, key, return_behavior=False):
        data_key = (self.target_unit_table & key).fetch("data_key")
        dat = dataloaders["train"][data_key].dataset
        image_class, image_id = (self.base_image_table & key).fetch1(
            "image_class", "image_id"
        )
        if return_behavior:
            behavior_keys = get_image_data_from_dataset(
                dat, image_class, image_id, return_behavior=True
            )
        else:
            responses = get_image_data_from_dataset(
                dat, image_class, image_id, return_behavior=False
            )
        return responses if return_behavior is False else behavior_keys

    def get_real_behavior(self, key):
        return self.get_neuronal_responses(key, return_behavior=True)

    def get_original_image(self, key):
        image = (self.base_image_table & key).fetch1("image")
        print(image.shape)
        image = preprocess_img_for_reconstruction(
            image, img_size=self.reconstruction_size
        )
        print(image.shape)
        return image

    def make(self, key):
        dataloaders, model = self.model_loader.load(key=key)
        seed = (self.seed_table() & key).fetch1("mei_seed")
        recon_type = (self.recon_type_table & key).fetch1("recon_type")
        image = self.get_original_image(key)
        behavior, kwargs = get_behavior_from_method_config(
            (self.method_table & key).fetch1("method_config")
        )

        responses = (
            self.get_neuronal_responses(dataloaders=dataloaders, key=key)
            if recon_type == "neurons"
            else self.get_model_responses(
                model=model,
                key=key,
                image=image,
                behavior=behavior,
                **kwargs,
            )
        )

        target_fn = (self.target_fn_table & key).get_target_fn(responses=responses)
        unit_ids, data_key = (self.target_unit_table & key).fetch1(
            "unit_ids", "data_key"
        )

        output_selected_model = self.selector_table().get_output_selected_model(
            model=model,
            target_fn=target_fn,
            unit_ids=unit_ids,
            data_key=data_key,
        )
        mei_entity = self.method_table().generate_mei(
            dataloaders, output_selected_model, key, seed
        )
        self._insert_mei(mei_entity)
