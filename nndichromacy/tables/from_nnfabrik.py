import datajoint as dj

from nnfabrik.templates.trained_model import TrainedModelBase, DataInfoBase
from nnfabrik.utility.dj_helpers import gitlog, make_hash
from nnfabrik.builder import resolve_data, get_data
from nnfabrik.utility.dj_helpers import CustomSchema
from nnfabrik.main import Dataset, Trainer, Model, Fabrikant, Seed, my_nnfabrik

import os
import pickle
from ..utility.dj_helpers import get_default_args
from ..datasets.mouse_loaders import static_loader
from .templates import (
    ScoringBase,
    MeasuresBase,
    SummaryMeasuresBase,
    SummaryScoringBase,
)

schema = CustomSchema(dj.config.get("nnfabrik.schema_name", "nnfabrik_core"))




if not 'stores' in dj.config:
    dj.config['stores'] = {}
dj.config['stores']['minio'] = {  # store in s3
        'protocol': 's3',
        'endpoint': os.environ.get('MINIO_ENDPOINT', 'DUMMY_ENDPOINT'),
        'bucket': 'nnfabrik',
        'location': 'dj-store',
        'access_key': os.environ.get('MINIO_ACCESS_KEY', 'FAKEKEY'),
        'secret_key': os.environ.get('MINIO_SECRET_KEY', 'FAKEKEY'),
        'secure': True,
    }


@schema
class DataInfo(DataInfoBase):
    dataset_table = Dataset
    user_table = Fabrikant

    def create_stats_files(self, key=None, path=None):

        if key == None:
            key = self.fetch("KEY")

            for restr in key:
                dataset_config = (self.dataset_table & restr).fetch1("dataset_config")
                image_cache_path = dataset_config.get("image_cache_path", None)
                if image_cache_path is None:
                    raise ValueError(
                        "The argument image_cache_path has to be specified in the dataset_config in order "
                        "to create the DataInfo"
                    )

                image_cache_path = image_cache_path.split("individual")[0]
                default_args = get_default_args(
                    resolve_data((self.dataset_table & restr).fetch1("dataset_fn"))
                )
                default_args.update(dataset_config)
                stats_filename = make_hash(default_args)
                stats_path = os.path.join(
                    path if path is not None else image_cache_path,
                    "statistics/",
                    stats_filename,
                )

                if not os.path.exists(stats_path):
                    data_info = (self & restr).fetch1("data_info")

                    with open(stats_path, "wb") as pkl:
                        pickle.dump(data_info, pkl)


@schema
class TrainedModel(TrainedModelBase):
    table_comment = "Trained models"
    data_info_table = DataInfo
    storage = "minio"

    model_table = Model
    dataset_table = Dataset
    trainer_table = Trainer
    seed_table = Seed
    user_table = Fabrikant


my_nnfabrik_1 = my_nnfabrik(
    dj.config.get("nnfabrik.hypersearch_schema", "nnfabrik_core"),
    use_common_fabrikant=False,
)


@my_nnfabrik_1.schema
class TrainedHyperModel(TrainedModelBase):
    nnfabrik = my_nnfabrik_1


class ScoringTable(ScoringBase):
    """
    Overwrites the nnfabriks scoring template, to make it handle mouse repeat-dataloaders.
    """

    dataloader_function_kwargs = {}

    def get_repeats_dataloaders(self, key=None, **kwargs):
        if key is None:
            key = self.fetch1("KEY")

        dataset_fn, dataset_config = (self.dataset_table() & key).fn_config
        dataset_config.update(kwargs)
        dataset_config["return_test_sampler"] = True
        dataloaders = get_data(dataset_fn, dataset_config)
        return dataloaders

    def get_model(self, key=None):
        if self.model_cache is None:
            model = self.trainedmodel_table().load_model(
                key=key, include_state_dict=True, include_dataloader=False
            )
        else:
            model = self.model_cache.load(
                key=key, include_state_dict=True, include_dataloader=False
            )
        model.eval()
        model.to("cuda")
        return model

    def make(self, key):

        dataloaders = (
            self.get_repeats_dataloaders(key=key, **self.dataloader_function_kwargs)
            if self.measure_dataset == "test"
            else self.get_dataloaders(key=key)
        )
        model = self.get_model(key=key)
        unit_measures_dict = self.measure_function(
            model=model,
            dataloaders=dataloaders,
            device="cuda",
            as_dict=True,
            per_neuron=True,
            **self.function_kwargs
        )

        key[self.measure_attribute] = self.get_avg_of_unit_dict(unit_measures_dict)
        self.insert1(key, ignore_extra_fields=True)
        self.insert_unit_measures(key=key, unit_measures_dict=unit_measures_dict)


class SummaryScoringTable(ScoringTable):
    """
    A template scoring table with the same logic as ScoringBase, but for scores that do not have unit scores, but
    an overall score per model only.
    """

    unit_table = None
    Units = None

    def make(self, key):

        dataloaders = (
            self.get_repeats_dataloaders(key=key, **self.dataloader_function_kwargs)
            if self.measure_dataset == "test"
            else self.get_dataloaders(key=key)
        )
        model = self.get_model(key=key)
        key[self.measure_attribute] = self.measure_function(
            model=model, dataloaders=dataloaders, device="cuda", **self.function_kwargs
        )
        self.insert1(key, ignore_extra_fields=True)


class MeasuresTable(MeasuresBase, ScoringTable):
    """
    Overwrites the nnfabriks scoring template, to make it handle mouse repeat-dataloaders.
    """

    dataloader_function_kwargs = {}

    def make(self, key):

        dataloaders = (
            ScoringTable.get_repeats_dataloaders(
                self, key=key, **self.dataloader_function_kwargs
            )
            if self.measure_dataset == "test"
            else self.get_dataloaders(key=key)
        )
        unit_measures_dict = self.measure_function(
            dataloaders=dataloaders,
            as_dict=True,
            per_neuron=True,
            **self.function_kwargs
        )

        key[self.measure_attribute] = self.get_avg_of_unit_dict(unit_measures_dict)
        self.insert1(key, ignore_extra_fields=True)
        self.insert_unit_measures(key=key, unit_measures_dict=unit_measures_dict)
