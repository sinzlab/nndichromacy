"""This module contains mix-ins for the main tables and table templates."""

from __future__ import annotations
import os
import tempfile
from typing import Callable, Iterable, Mapping, Optional, Tuple, Dict, Any
from string import ascii_letters
from random import choice

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from nnfabrik.utility.dj_helpers import make_hash

from mei import integration

import warnings
from functools import partial

import datajoint as dj
from nnfabrik.main import Dataset
from nnfabrik.utility.dj_helpers import CustomSchema
from nnfabrik.builder import resolve_fn
from mei.main import MEISeed,MEIMethod
from mei.modules import EnsembleModel, ConstrainedOutputModel
from .from_mei import TrainedEnsembleModel
from .from_mei import MEISelector
from .from_mei import MEI
from .from_nnfabrik import TrainedModel
from mei import mixins
import os

fetch_download_path = os.environ.get('FETCH_DOWNLOAD_PATH', '/data/fetched_from_attach')

schema = CustomSchema(dj.config.get("nnfabrik.schema_name", "nnfabrik_core"))
resolve_target_fn = partial(resolve_fn, default_base="targets")


Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]

@schema
class SurrMEIRelateHash(dj.Manual):
    definition = """
    # contains ensemble ids
    inner_ensemble_hash                   : char(32)      # the ensemble hash for linear model
    inner_method_hash                       : char(32)
    src_method_fn                         : char(32)      
    ---
    hash_comment        = ''    : varchar(256)  # a short comment describing the MEI version
    """

class SurrMEITemplateMixin:
    definition = """
    # contains maximally exciting images (MEIs)
    -> self.mei_table.proj(src_method_fn='method_fn', inner_ensemble_hash='ensemble_hash', inner_method_hash='method_hash', inner_mei_seed='mei_seed')
    -> self.method_table
    -> self.selector_table
    -> self.trained_model_table
    -> self.seed_table

    ---
    mei             : attach@minio  # the MEI as a tensor
    score               : float         # some score depending on the used method function
    output              : attach@minio  # object returned by the method function
    """

    @property
    def key_source(self):
        return super().key_source & SurrMEIRelateHash #& 'method_fn like "%ring%"'

    method_table = None
    trained_model_table = None
    mei_table = None
    selector_table = None
    seed_table = None
    surr_mei_relate_hash_table = None
    model_loader_class = integration.ModelLoader
    save = staticmethod(torch.save)
    get_temp_dir = tempfile.TemporaryDirectory

    insert1: Callable[[Mapping], None]

    def __init__(self, *args, cache_size_limit: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_loader = self.model_loader_class(self.trained_model_table, cache_size_limit=cache_size_limit)

    def make(self, key: Key) -> None:
        dataloaders, model = self.model_loader.load(key=key)
        seed = (self.seed_table() & key).fetch1("mei_seed")
        output_selected_model = self.selector_table().get_output_selected_model(model, key)
        mei_entity = self.method_table().generate_mei(dataloaders, output_selected_model, key, seed)
        self._insert_mei(mei_entity)

    def _insert_mei(self, mei_entity: Dict[str, Any]) -> None:
        """Saves the MEI to a temporary directory and inserts the prepared entity into the table."""
        with self.get_temp_dir() as temp_dir:
            for name in ("mei", "output"):
                self._save_to_disk(mei_entity, temp_dir, name)
            self.insert1(mei_entity, ignore_extra_fields=True)

    def _save_to_disk(self, mei_entity: Dict[str, Any], temp_dir: str, name: str) -> None:
        data = mei_entity.pop(name)
        filename = name + "_" + self._create_random_filename() + ".pth.tar"
        filepath = os.path.join(temp_dir, filename)
        self.save(data, filepath)
        mei_entity[name] = filepath

    @staticmethod
    def _create_random_filename(length: Optional[int] = 32) -> str:
        return "".join(choice(ascii_letters) for _ in range(length))

###------------------------(nndichromicy from_mei)--------------------------------


@schema
class SurrMEI(SurrMEITemplateMixin, dj.Computed):
    """MEI table template.

    To create a functional "SurrMEI" table, create a new class that inherits from this template and decorate it with your
    preferred Datajoint schema. Next assign your trained model (or trained ensemble model) and your selector table to
    the class variables called "trained_model_table" and "selector_table". By default, the created table will point to
    the "MEIMethod" table in the Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting
    the class attribute called "method_table".
    """

    trained_model_table = TrainedEnsembleModel
    method_table = MEIMethod
    seed_table = MEISeed
    mei_table = MEI # need normal mei obtain mask
    surr_mei_relate_hash_table = SurrMEIRelateHash
    selector_table = MEISelector

