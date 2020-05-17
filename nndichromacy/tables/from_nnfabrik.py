import datajoint as dj
from nnfabrik.template import TrainedModelBase, ScoringBase, MeasuresBase
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from nnfabrik.utility.dj_helpers import gitlog, make_hash
from nnfabrik.template import DataInfoBase, ScoringBase
from nnfabrik.builder import resolve_data
from nnfabrik.utility.dj_helpers import CustomSchema
import os
import pickle
from ..utility.dj_helpers import get_default_args

schema = CustomSchema(dj.config.get('schema_name', 'nnfabrik_core'))


@schema
class DataInfo(DataInfoBase):

    def create_stats_files(self, key=None, path=None):

        if key == None:
            key = self.fetch('KEY')

            for restr in key:
                dataset_config = (self.dataset_table & restr).fetch1("dataset_config")
                image_cache_path = dataset_config.get("image_cache_path", None)
                if image_cache_path is None:
                    raise ValueError("The argument image_cache_path has to be specified in the dataset_config in order "
                                     "to create the DataInfo")

                image_cache_path = image_cache_path.split('individual')[0]
                default_args = get_default_args(resolve_data((self.dataset_table & restr).fetch1("dataset_fn")))
                default_args.update(dataset_config)
                stats_filename = make_hash(default_args)
                stats_path = os.path.join(path if path is not None else image_cache_path, 'statistics/', stats_filename)

                if not os.path.exists(stats_path):
                    data_info = (self & restr).fetch1("data_info")

                    with open(stats_path, "wb") as pkl:
                        pickle.dump(data_info, pkl)


@schema
class TrainedModel(TrainedModelBase):
    table_comment = "Trained models"
    data_info_table = DataInfo


class ScoringBaseNeuronType(ScoringBase):
    """
    A class that modifies the the scoring template from nnfabrik to reflect the changed primary attributes of the Units
    table.
    """
    def insert_unit_measures(self, key, unit_measures_dict):
        key = key.copy()
        for data_key, unit_scores in unit_measures_dict.items():
            for unit_index, unit_score in enumerate(unit_scores):
                if "unit_id" in key: key.pop("unit_id")
                if "data_key" in key: key.pop("data_key")
                if "unit_type" in key: key.pop("unit_type")
                neuron_key = dict(unit_index=unit_index, data_key=data_key)
                unit_id = ((self.unit_table & key) & neuron_key).fetch1("unit_id")
                unit_type = ((self.unit_table & key) & neuron_key).fetch1("unit_type")
                key["unit_id"] = unit_id
                key["unit_type"] = unit_type
                key["unit_{}".format(self.measure_attribute)] = unit_score
                key["data_key"] = data_key
                self.Units.insert1(key, ignore_extra_fields=True)


class MeasuresBaseNeuronType(MeasuresBase):
    """
    A class that modifies the the scoring template from nnfabrik to reflect the changed primary attributes of the Units
    table.
    """
    def insert_unit_measures(self, key, unit_measures_dict):
        key = key.copy()
        for data_key, unit_scores in unit_measures_dict.items():
            for unit_index, unit_score in enumerate(unit_scores):
                if "unit_id" in key: key.pop("unit_id")
                if "data_key" in key: key.pop("data_key")
                if "unit_type" in key: key.pop("unit_type")
                neuron_key = dict(unit_index=unit_index, data_key=data_key)
                unit_id = ((self.unit_table & key) & neuron_key).fetch1("unit_id")
                unit_type = ((self.unit_table & key) & neuron_key).fetch1("unit_type")
                key["unit_id"] = unit_id
                key["unit_type"] = unit_type
                key["unit_{}".format(self.measure_attribute)] = unit_score
                key["data_key"] = data_key
                self.Units.insert1(key, ignore_extra_fields=True)

