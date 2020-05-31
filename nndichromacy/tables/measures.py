import datajoint as dj
from nnfabrik.template import TrainedModelBase
import tempfile
import torch
import os
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from nnfabrik.utility.dj_helpers import gitlog, make_hash
import numpy as np
from ..utility.measures import get_oracles, get_oracles_corrected, get_repeats, get_FEV, get_explainable_var, get_correlations, get_poisson_loss, get_avg_correlations, get_predictions, get_targets
from .from_nnfabrik import TrainedModel, ScoringTable
from .from_mei import MEISelector, TrainedEnsembleModel
from .utility import DataCache, TrainedModelCache, EnsembleModelCache
from nnfabrik.utility.dj_helpers import CustomSchema
from nnfabrik.template import ScoringBase, SummaryScoringBase, MeasuresBase

schema = CustomSchema(dj.config.get('schema_name', 'nnfabrik_core'))


@schema
class ExplainableVar(MeasuresBase):
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_explainable_var)
    measure_dataset = "test"
    measure_attribute = "fev"
    data_cache = DataCache


@schema
class OracleScore(MeasuresBase):
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_oracles)
    measure_dataset = "test"
    measure_attribute = "oracle_score"
    data_cache = DataCache

@schema
class OracleScoreCorrected(MeasuresBase):
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_oracles_corrected)
    measure_dataset = "test"
    measure_attribute = "oracle_score_corrected"
    data_cache = DataCache


@schema
class ModelRFSize(SummaryMeasuresBase):
    dataset_table = Model
    unit_table = None
    measure_function = staticmethod(get_model_rf_size)
    measure_attribute = "model_rf_size"

    def make(self, key):
        model_config = (self.dataset_table & key).fetch1("model_config")
        default_args = get_default_args(resolve_model((self.dataset_table & key).fetch1("model_fn")))
        default_args.update(model_config)
        key[self.measure_attribute] = self.measure_function(default_args)
        self.insert1(key, ignore_extra_fields=True)
