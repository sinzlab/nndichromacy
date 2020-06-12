import datajoint as dj
from nnfabrik.template import TrainedModelBase
import tempfile
import torch
import os
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from nnfabrik.utility.dj_helpers import gitlog, make_hash
import numpy as np
from ..utility.measures import get_oracles, get_repeats, get_FEV, get_explainable_var, get_correlations, \
    get_poisson_loss, get_avg_correlations, get_predictions, get_targets, get_fraction_oracles
from .from_nnfabrik import TrainedModel, ScoringTable, SummaryScoringTable
from .from_mei import MEISelector, TrainedEnsembleModel
from .utility import DataCache, TrainedModelCache, EnsembleModelCache
from nnfabrik.utility.dj_helpers import CustomSchema

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
class TestCorrelation_BlueTestSet(ScoringTable):
    trainedmodel_table = TrainedModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test"
    measure_attribute = "test_correlation_blue"
    data_cache = DataCache
    dataloader_function_kwargs = dict(image_condition='imagenet_v2_rgb_blue_only')


@schema
class TestCorrelation_GreenTestSet(ScoringTable):
    trainedmodel_table = TrainedModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test"
    measure_attribute = "test_correlation_green"
    data_cache = DataCache
    dataloader_function_kwargs = dict(image_condition='imagenet_v2_rgb_green_only')


@schema
class TestCorrelation_DependentTestSet(ScoringTable):
    trainedmodel_table = TrainedModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test"
    measure_attribute = "test_correlation_dependent"
    data_cache = DataCache
    dataloader_function_kwargs = dict(image_condition='imagenet_v2_rgb_gb_g_biased_correlated')


@schema
class TestCorrelationEnsemble(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test"
    measure_attribute = "avg_correlation"
    data_cache = DataCache
    model_cache = EnsembleModelCache



@schema
class CorrelationToAvergeEnsemble(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_correlations)
    measure_dataset = "test"
    measure_attribute = "avg_correlation"
    data_cache = None
    model_cache = None


##### ============ Summary Scores ============ #####

@schema
class FractionOracleJackknife(SummaryScoringTable):
    trainedmodel_table = TrainedModel
    measure_function = staticmethod(get_fraction_oracles)
    measure_dataset = "test"
    measure_attribute = "fraction_oracle_jackknife"
    data_cache = None
    model_cache = None



@schema
class FractionOracleJackknifeEnsemble(SummaryScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    measure_function = staticmethod(get_fraction_oracles)
    measure_dataset = "test"
    measure_attribute = "fraction_oracle_jackknife"
    data_cache = None
    model_cache = None


