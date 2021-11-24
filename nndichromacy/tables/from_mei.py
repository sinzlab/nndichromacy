from __future__ import annotations
from typing import Dict, Any

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]

import warnings
from functools import partial

import datajoint as dj
from nnfabrik.main import Dataset
from nnfabrik.utility.dj_helpers import CustomSchema, make_hash, cleanup_numpy_scalar
from nnfabrik.builder import resolve_fn
from mei import mixins
from mei.main import MEISeed
from mei.modules import ConstrainedOutputModel
from .from_nnfabrik import TrainedModel

schema = CustomSchema(dj.config.get("nnfabrik.schema_name", "nnfabrik_core"))
resolve_target_fn = partial(resolve_fn, default_base="targets")
resolve_mask_fn = partial(resolve_fn, default_base="masks")


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
    optional_names = (
        "initial",
        "optimizer",
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
                if "kwargs" in v:
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
class ThresholdMEIMaskConfig(dj.Manual):
    definition = """
    mask_fn:       varchar(64)
    mask_hash:     varchar(64)
    ---
    zscore_threshold:  float
    closing_iters:     int
    gauss_sigma:       float
    """

    def add_entry(self, mask_fn, mask_config, skip_duplicates=False):

        try:
            resolve_mask_fn(mask_fn)
        except (NameError, TypeError) as e:
            warnings.warn(str(e) + "\nTable entry rejected")
            return

        mask_hash = make_hash(mask_config)
        key = dict(
            mask_fn=mask_fn,
            mask_hash=mask_hash,
            **mask_config,
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

    @property
    def fn_config(self):
        mask_fn, zscore_threshold, closing_iters, gauss_sigma = self.fetch1("mask_fn", "zscore_threshold", "closing_iters", "gauss_sigma")
        mask_fn = resolve_mask_fn(mask_fn)
        mask_config = dict(zscore_threshold=zscore_threshold,
                      closing_iters=closing_iters,
                      gauss_sigma=gauss_sigma)
        return mask_fn, mask_config