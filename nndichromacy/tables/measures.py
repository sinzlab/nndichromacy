import datajoint as dj

from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant

from ..utility.measures import get_oracles, get_model_rf_size, get_oracles_corrected, get_repeats, get_FEV, \
    get_explainable_var, get_correlations, get_poisson_loss, get_avg_correlations, get_predictions, get_targets, get_avg_firing, get_SNR

from .from_mei import MEISelector
from .from_nnfabrik import MeasuresTable, SummaryMeasuresBase

from . import DataCache, TrainedModelCache, EnsembleModelCache
from nnfabrik.builder import resolve_model
from nnfabrik.utility.dj_helpers import CustomSchema
from ..utility.dj_helpers import get_default_args

schema = CustomSchema(dj.config.get('nnfabrik.schema_name', 'nnfabrik_core'))


@schema
class ExplainableVar(MeasuresTable):
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_explainable_var)
    measure_dataset = "test"
    measure_attribute = "fev"
    data_cache = DataCache


@schema
class SignalToNoiseRatio(MeasuresTable):
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_SNR)
    measure_dataset = "test"
    measure_attribute = "snr"
    data_cache = DataCache


@schema
class JackknifeOracle(MeasuresTable):
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_oracles)
    measure_dataset = "test"
    measure_attribute = "jackknife_oracle"
    data_cache = DataCache


@schema
class OracleCorrelation(MeasuresTable):
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_oracles_corrected)
    measure_dataset = "test"
    measure_attribute = "oracle_correlation"
    data_cache = DataCache


@schema
class OracleCorrelationBlueSet(MeasuresTable):
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_oracles_corrected)
    measure_dataset = "test"
    measure_attribute = "oracle_correlation_blue"
    data_cache = DataCache
    dataloader_function_kwargs = dict(image_condition='imagenet_v2_rgb_blue_only')


@schema
class OracleCorrelationGreenSet(MeasuresTable):
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_oracles_corrected)
    measure_dataset = "test"
    measure_attribute = "oracle_correlation_green"
    data_cache = DataCache
    dataloader_function_kwargs = dict(image_condition='imagenet_v2_rgb_green_only')


@schema
class OracleCorrelationDepSet(MeasuresTable):
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_oracles_corrected)
    measure_dataset = "test"
    measure_attribute = "oracle_correlation_dependent"
    data_cache = DataCache
    dataloader_function_kwargs = dict(image_condition='imagenet_v2_rgb')


@schema
class AvgFiringRateTest(MeasuresTable):
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_firing)
    measure_dataset = "test"
    measure_attribute = "avg_firing_test"
    data_cache = DataCache


@schema
class AvgFiringRateTestDepSet(MeasuresTable):
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_firing)
    measure_dataset = "test"
    measure_attribute = "avg_firing_test_dependent"
    data_cache = DataCache
    dataloader_function_kwargs = dict(image_condition='imagenet_v2_rgb')


@schema
class AvgFiringRateTestGreenSet(MeasuresTable):
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_firing)
    measure_dataset = "test"
    measure_attribute = "avg_firing_test_green"
    data_cache = DataCache
    dataloader_function_kwargs = dict(image_condition='imagenet_v2_rgb_green_only')


@schema
class AvgFiringRateTestBlueSet(MeasuresTable):
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_firing)
    measure_dataset = "test"
    measure_attribute = "avg_firing_test_blue"
    data_cache = DataCache
    dataloader_function_kwargs = dict(image_condition='imagenet_v2_rgb_blue_only')


##### ============ Summary Scores ============ #####


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


