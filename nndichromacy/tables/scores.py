import datajoint as dj
from nnfabrik.template import TrainedModelBase
import tempfile
import torch
import os
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from nnfabrik.utility.dj_helpers import gitlog, make_hash
import numpy as np
from ..utility.measures import get_oracles, get_repeats, get_FEV, get_explainable_var, get_correlations, get_poisson_loss, get_avg_correlations, get_predictions, get_targets
from .from_nnfabrik import TrainedModel, ScoringTable
from .from_mei import MEISelector, TrainedEnsembleModel
from .utility import DataCache, TrainedModelCache, EnsembleModelCache
from nnfabrik.utility.dj_helpers import CustomSchema
from nnfabrik.template import ScoringBase, SummaryScoringBase

schema = CustomSchema(dj.config.get('schema_name', 'nnfabrik_core'))


@schema
class CorrelationToAverge(ScoringTable):
    trainedmodel_table = TrainedModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_correlations)
    measure_dataset = "test"
    measure_attribute = "avg_correlation"
    data_cache = DataCache
    model_cache = TrainedModelCache


@schema
class TestCorrelation(ScoringTable):
    trainedmodel_table = TrainedModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test"
    measure_attribute = "test_correlation"
    data_cache = DataCache
    model_cache = TrainedModelCache


@schema
class TestCorrelationEnsemble(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_correlations)
    measure_dataset = "test"
    measure_attribute = "avg_correlation"
    data_cache = DataCache
    model_cache = EnsembleModelCache
