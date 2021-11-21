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
from mei.modules import EnsembleModel, ConstrainedOutputModel #, ContrastiveOutputModel
from .tables.from_mei import TrainedEnsembleModel
from .tables.from_nnfabrik import TrainedModel
from mei import mixins

schema = CustomSchema(dj.config.get("nnfabrik.schema_name", "nnfabrik_core"))
resolve_target_fn = partial(resolve_fn, default_base="targets")


Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]

class ContrastiveOutputModel(Module):
    """A model that has its output constrained.

    Attributes:
        model: A PyTorch module.
        constraint: An integer representing the index of a neuron in the model's output. Only the value corresponding
            to that index will be returned.
        target_fn: Callable, that gets as an input the constrained output of the model.
        forward_kwargs: A dictionary containing keyword arguments that will be passed to the model every time it is
            called. Optional.
    """

    def __init__(self, model1: Module, model2: Module, constraint: int, target_fn=None, forward_kwargs: Dict[str, Any] = None):
        """Initializes ConstrainedOutputModel."""
        super().__init__()
        if target_fn is None:
            target_fn = lambda x: x
        self.model1 = model1
        self.model2 = model2
        self.constraint = constraint
        self.forward_kwargs = forward_kwargs if forward_kwargs else dict()
        self.target_fn = target_fn

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Computes the constrained output of the model.

        Args:
            x: A tensor representing the input to the model.
            *args: Additional arguments will be passed to the model.
            **kwargs: Additional keyword arguments will be passed to the model.

        Returns:
            A tensor representing the constrained output of the model.
        """
        output1 = self.model1(x, *args, **self.forward_kwargs, **kwargs)
        output2 = self.model2(x, *args, **self.forward_kwargs, **kwargs)
        contrast_output=output1-output2
        return self.target_fn(contrast_output[:, self.constraint])

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self.model1},{self.model2} ,{self.constraint}, forward_kwargs={self.forward_kwargs})"

class CSRFV1SelectorTemplateMixin:
    definition = """
    # contains information that can be used to map a neuron's id to its corresponding integer position in the output of
    # the model. 
    -> self.dataset_table
    neuron_id       : smallint unsigned # unique neuron identifier
    ---
    neuron_position : smallint unsigned # integer position of the neuron in the model's output 
    session_id      : varchar(13)       # unique session identifier
    """

    dataset_table = None
    dataset_fn = "csrf_v1"
    constrained_output_model = ConstrainedOutputModel

    insert: Callable[[Iterable], None]
    __and__: Callable[[Mapping], CSRFV1SelectorTemplateMixin]
    fetch1: Callable

    @property
    def _key_source(self):
        return self.dataset_table() & dict(dataset_fn=self.dataset_fn)

    def make(self, key: Key, get_mappings: Callable = integration.get_mappings) -> None:
        dataset_config = (self.dataset_table() & key).fetch1("dataset_config")
        mappings = get_mappings(dataset_config, key)
        self.insert(mappings)

    def get_output_selected_model(self, model: Module, key: Key) -> constrained_output_model:
        neuron_pos, session_id = (self & key).fetch1("neuron_position", "session_id")
        return self.constrained_output_model(model, neuron_pos, forward_kwargs=dict(data_key=session_id))


class ContrastiveMEITemplateMixin:
    definition = """
    # contains maximally exciting images (MEIs)
    -> self.method_table
    -> self.trained_model_table
    -> self.contrastive_model_table
    -> self.selector_table
    -> self.seed_table
    ---
    mei                 : attach@minio  # the MEI as a tensor
    score               : float         # some score depending on the used method function
    output              : attach@minio  # object returned by the method function
    """

    trained_model_table = None
    contrastive_model_table = None
    selector_table = None
    method_table = None
    seed_table = None
    model_loader_class = integration.ModelLoader
    save = staticmethod(torch.save)
    get_temp_dir = tempfile.TemporaryDirectory

    insert1: Callable[[Mapping], None]

    def __init__(self, *args, cache_size_limit: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_loader = self.model_loader_class(self.trained_model_table, cache_size_limit=cache_size_limit)
        #self.model_loader_linear = self.model_loader_class(self.contrastive_model_table, cache_size_limit=cache_size_limit)
    def make(self, key: Key) -> None:
        dataloaders, model_nonlinear = self.model_loader.load(key=key)
        #print(key)
        contrast_hash=(self.contrastive_model_table() & key).fetch1('contrast_hash')
        new_key=(self.trained_model_table & dict(ensemble_hash=contrast_hash) ).fetch("KEY")[0]
        #print(new_key)
        #new_key=dj.AndList([dict(ensemble_hash=contrast_hash),
        _, model_linear = self.model_loader.load(key=new_key)
        seed = (self.seed_table() & key).fetch1("mei_seed")
        output_selected_model = self.selector_table().get_output_selected_model(model_nonlinear,model_linear, key)
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

class ContrastiveSelectorTemplate(dj.Computed):

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

    constrained_output_model = ContrastiveOutputModel

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
        self, model1: Module, model2:Module, key: Key
    ) -> constrained_output_model:
        unit_index, data_key = (self & key).fetch1("unit_index", "data_key")
        return self.constrained_output_model(
            model1,model2, unit_index, forward_kwargs=dict(data_key=data_key)
        )


@schema
class ContrastiveMEISelector(ContrastiveSelectorTemplate):
    dataset_table = Dataset

'''@schema
class ContrastiveEnsembleModel2(mixins.TrainedEnsembleModelTemplateMixin, dj.Manual):
    dataset_table = Dataset
    trained_model_table = TrainedModel

    class Member(mixins.TrainedEnsembleModelTemplateMixin.Member, dj.Part):
        """Member table template."""
        pass
'''
@schema
class ContrastiveEnsembleTest(dj.Manual):
    definition = """
    # contains ensemble ids
    contrast_hash                   : char(32)      # the hash of the contrative ensemble
    ---
    contrastive_ensemble_comment        = ''    : varchar(256)  # a short comment describing the ensemble
    """

@schema
class ContrastiveMEI(ContrastiveMEITemplateMixin, dj.Computed):
    """MEI table template.

    To create a functional "MEI" table, create a new class that inherits from this template and decorate it with your
    preferred Datajoint schema. Next assign your trained model (or trained ensemble model) and your selector table to
    the class variables called "trained_model_table" and "selector_table". By default, the created table will point to
    the "MEIMethod" table in the Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting
    the class attribute called "method_table".
    """

    trained_model_table = TrainedEnsembleModel
    contrastive_model_table = ContrastiveEnsembleTest # could be manual table
    selector_table = ContrastiveMEISelector
    method_table = MEIMethod
    seed_table = MEISeed


