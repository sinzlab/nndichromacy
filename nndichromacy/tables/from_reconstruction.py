from __future__ import annotations
from typing import Dict, Any

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]

import warnings
from functools import partial
from typing import Dict, Any, Callable, List

import datajoint as dj
from nnfabrik.main import Dataset, Trainer, Model, Fabrikant, Seed, my_nnfabrik
from nnfabrik.utility.dj_helpers import CustomSchema, make_hash, cleanup_numpy_scalar
from nnfabrik.builder import resolve_fn

from mei.modules import ConstrainedOutputModel

from nndichromacy.tables import TrainedEnsembleModel
from nndichromacy.tables.from_mei import schema, Dataloaders, Key, resolve_target_fn
from .utils import (
    get_image_data_from_dataset,
    preprocess_img_for_reconstruction,
    get_initial_image,
    process_image,
)
from mei import mixins
from .from_mei import MEISeed
from dataport.bcm.color_mei.schema import StaticImage

schema = CustomSchema(dj.config.get("nnfabrik.schema_name", "nnfabrik_core"))
resolve_target_fn = partial(resolve_fn, default_base="targets")


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
        if "image_repeat" in method_config:
            print("... updating method config ...")
            dataloaders = (
                Reconstruction().trained_model_table.dataset_table() & key
            ).get_dataloader()

            responses, behavior, image = Reconstruction().get_neuronal_responses(
                dataloaders=dataloaders,
                key=key,
                return_behavior=True,
                method_config=method_config,
                return_image=True,
            )
            behavior_list = [*behavior[0].cpu().numpy().squeeze()]
            eye_pos = [*behavior[1].cpu().numpy().squeeze()]

            method_config["initial"]["kwargs"]["selected_values"] = behavior_list
            if "model_forward_kwargs" not in method_config:
                method_config["model_forward_kwargs"] = {}

            method_config["model_forward_kwargs"]["eye_pos"] = np.array([eye_pos])
            method_config["model_forward_kwargs"]["behavior"] = np.array(
                [behavior_list]
            )
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
class ReconMethodExperiments(dj.Manual):
    definition = """
    experiment_name: varchar(100)                     # name of experiment
    ---
    -> Fabrikant.proj(experiment_fabrikant='fabrikant_name')
    experiment_comment='': varchar(2000)              # short description 
    experiment_ts=CURRENT_TIMESTAMP:   timestamp      # UTZ timestamp at time of insertion
    """

    class Restrictions(dj.Part):
        definition = """
        # This table contains the corresponding hashes to filter out models which form the respective experiment
        -> master
        -> ReconMethod
        ---
        experiment_restriction_ts=CURRENT_TIMESTAMP:   timestamp      # UTZ timestamp at time of insertion
    """


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
    storage = "minio"
    database = ""  # hack to supress DJ error

    class Responses(dj.Part):
        @property
        def definition(self):
            definition = """
            # Contains the models state dict, stored externally.
            -> master
            ---
            original_responses:                 attach@{storage}
            reconstructed_responses:            attach@{storage}
            """.format(
                storage=self._master.storage
            )
            return definition

    def get_model_responses(
        self,
        model,
        key,
        image,
        device="cuda",
        forward_kwargs=None,
        constraint=None,
    ):
        model.eval()
        model.to(device)
        forward_kwargs = dict() if forward_kwargs is None else forward_kwargs
        with torch.no_grad():
            responses = model(
                image.to(device),
                data_key=key["data_key"],
                **forward_kwargs,
            )
        return (
            responses
            if constraint is None or len(constraint) == 0
            else responses[:, constraint]
        )

    def get_neuronal_responses(
        self,
        dataloaders,
        key,
        method_config,
        return_behavior=False,
        return_image=False,
    ):
        data_key = (self.target_unit_table & key).fetch1("data_key")
        dat = dataloaders["train"][data_key].dataset
        image_class, image_id = (self.base_image_table & key).fetch1(
            "image_class", "image_id"
        )
        image_repeat = method_config.get("image_repeat", None)
        if return_behavior:
            behavior_keys = get_image_data_from_dataset(
                dat,
                image_class,
                image_id,
                return_behavior=True,
                image_repeat=image_repeat,
            )

        if return_image:
            image = get_image_data_from_dataset(
                dat,
                image_class,
                image_id,
                return_behavior=False,
                image_repeat=image_repeat,
                return_image=return_image,
            )

        responses = get_image_data_from_dataset(
            dat,
            image_class,
            image_id,
            return_behavior=False,
            image_repeat=image_repeat,
        )

        ret = [responses]
        if return_behavior:
            ret.append(behavior_keys)
        if return_image:
            ret.append(image)
        return ret

    def get_real_behavior(self, key):
        return self.get_neuronal_responses(key, return_behavior=True)

    def get_original_image(self, key, img_statistics, dataloaders):
        image = (self.base_image_table & key).fetch1("image")

        image = preprocess_img_for_reconstruction(
            image,
            img_size=self.reconstruction_size,
            img_statistics=img_statistics,
            dataloaders=dataloaders,
        )
        return image

    def get_dataset_statistics(self, key, dataloaders):
        dataset_config = (self.trained_model_table.dataset_table & key).fetch1(
            "dataset_config"
        )
        mean, std = [dataset_config.get(i, None) for i in ["inputs_mean", "inputs_std"]]
        if mean is None:
            mean = dataloaders["train"][
                key["data_key"]
            ].dataset.statistics.images.all.mean
            std = dataloaders["train"][
                key["data_key"]
            ].dataset.statistics.images.all.std
        return mean, std

    def _insert_responses(self, response_entity: Dict[str, Any]) -> None:
        """Saves the MEI to a temporary directory and inserts the prepared entity into the table."""
        with self.get_temp_dir() as temp_dir:
            for name in ("original_responses", "reconstructed_responses"):
                self._save_to_disk(response_entity, temp_dir, name)
            self.Responses.insert1(response_entity, ignore_extra_fields=True)

    def make(self, key):
        dataloaders, model = self.model_loader.load(key=key)
        seed = (self.seed_table() & key).fetch1("mei_seed")
        method_config = (self.method_table & key).fetch1("method_config")
        recon_type = (self.recon_type_table & key).fetch1("recon_type")

        if recon_type == "neurons":
            # TODO: Make constraint based on unit IDs work (only works for model)
            responses, behavior, image = self.get_neuronal_responses(
                dataloaders=dataloaders,
                key=key,
                return_behavior=True,
                method_config=method_config,
                return_image=True,
            )
            behavior_list = [*behavior[0].cpu().numpy().squeeze()]
            eye_pos = [*behavior[1].cpu().numpy().squeeze()]

            method_config["initial"]["kwargs"]["selected_values"] = behavior_list
            if "model_forward_kwargs" not in method_config:
                method_config["model_forward_kwargs"] = {}

            method_config["model_forward_kwargs"]["eye_pos"] = np.array([eye_pos])
            method_config["model_forward_kwargs"]["behavior"] = np.array(
                [behavior_list]
            )

        else:
            img_mean, img_std = self.get_dataset_statistics(key, dataloaders)
            image = self.get_original_image(
                key, img_statistics=(img_mean, img_std), dataloaders=dataloaders
            )
            initial_img = get_initial_image(
                dataloaders=dataloaders,
                method_config=method_config,
                data_key=key["data_key"],
            )
            image = process_image(initial_img=initial_img, image=image)

            constraint = (self.target_unit_table & key).fetch1("unit_ids")

            responses = self.get_model_responses(
                model=model,
                key=key,
                image=image,
                forward_kwargs=method_config.get("model_forward_kwargs", None),
                constraint=constraint,
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

        reconstructed_image = mei_entity["mei"]
        # TODO: fix bug when using a constraint in the SelectorTable.
        reconstructed_responses = self.get_model_responses(
            model=model,
            key=key,
            image=reconstructed_image,
            forward_kwargs=method_config.get("model_forward_kwargs", None),
        )
        response_entity = dict(
            original_responses=responses,
            reconstructed_responses=reconstructed_responses,
        )

        self._insert_mei(mei_entity)
        mei_entity.update(response_entity)
        self._insert_responses(mei_entity)
