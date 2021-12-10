import datajoint as dj
from .mei_scores import MEINorm, MEINormBlue, MEINormGreen
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant

from ..utility.measures import (
    get_oracles,
    get_repeats,
    get_FEV,
    get_explainable_var,
    get_correlations,
    get_poisson_loss,
    get_avg_correlations,
    get_predictions,
    get_targets,
    get_fraction_oracles,
    get_mei_norm,
    get_mei_color_bias,
    get_conservative_avg_correlations,
    get_mei_michelson_contrast,
    get_r2er,
)
from .from_nnfabrik import TrainedModel, ScoringTable, SummaryScoringTable
from .from_mei import MEISelector, TrainedEnsembleModel
from .from_reconstruction import Reconstruction
from . import DataCache, TrainedModelCache, EnsembleModelCache
from nnfabrik.utility.dj_helpers import CustomSchema
from .from_mei import MEIScore

schema = CustomSchema(dj.config.get("nnfabrik.schema_name", "nnfabrik_core"))
fetch_download_path = "/data/fetched_from_attach/"


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
class R2er(ScoringTable):
    trainedmodel_table = TrainedModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_r2er)
    measure_dataset = "test"
    measure_attribute = "r2_er"
    data_cache = DataCache
    model_cache = TrainedModelCache


@schema
class CorrelationToAvgConservative(ScoringTable):
    trainedmodel_table = TrainedModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_conservative_avg_correlations)
    measure_dataset = "test"
    measure_attribute = "avg_correlation_cons"
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
class ValidationCorrelation(ScoringTable):
    trainedmodel_table = TrainedModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_correlations)
    measure_dataset = "validation"
    measure_attribute = "validation_correlation"
    data_cache = DataCache
    model_cache = TrainedModelCache


@schema
class TestCorrelationBlueTestSet(ScoringTable):
    trainedmodel_table = TrainedModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test"
    measure_attribute = "test_correlation_blue"
    data_cache = DataCache
    dataloader_function_kwargs = dict(image_condition="imagenet_v2_rgb_blue_only")


@schema
class TestCorrelationGreenTestSet(ScoringTable):
    trainedmodel_table = TrainedModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test"
    measure_attribute = "test_correlation_green"
    data_cache = DataCache
    dataloader_function_kwargs = dict(image_condition="imagenet_v2_rgb_green_only")


@schema
class TestCorrelationDependentTestSet(ScoringTable):
    trainedmodel_table = TrainedModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test"
    measure_attribute = "test_correlation_dependent"
    data_cache = DataCache
    dataloader_function_kwargs = dict(
        image_condition="imagenet_v2_rgb_gb_g_biased_correlated"
    )


@schema
class TestCorrelationDependenthighMSE(ScoringTable):
    trainedmodel_table = TrainedModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test"
    measure_attribute = "test_correlation_dependent"
    data_cache = DataCache
    dataloader_function_kwargs = dict(image_condition="imagenet_v2_rgb")


@schema
class FEVeScore(ScoringTable):
    trainedmodel_table = TrainedModel
    unit_table = MEISelector
    measure_function = staticmethod(get_FEV)
    measure_dataset = "test"
    measure_attribute = "feve"
    data_cache = DataCache
    model_cache = TrainedModelCache


@schema
class TestPoissonLoss(ScoringTable):
    dataset_table = Dataset
    trainedmodel_table = TrainedModel
    unit_table = MEISelector
    measure_function = staticmethod(get_poisson_loss)
    measure_dataset = "test"
    measure_attribute = "test_poissonloss"


##### ============ Ensemble Scores ============ #####


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
class ValidationCorrelationEnsemble(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_correlations)
    measure_dataset = "validation"
    measure_attribute = "avg_correlation"
    data_cache = DataCache
    model_cache = EnsembleModelCache


@schema
class TrainCorrelationEnsemble(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_correlations)
    measure_dataset = "train"
    measure_attribute = "avg_correlation"
    data_cache = DataCache
    model_cache = EnsembleModelCache


@schema
class TestPoissonLossEnsemble(ScoringTable):
    dataset_table = Dataset
    trainedmodel_table = TrainedEnsembleModel
    unit_table = MEISelector
    measure_function = staticmethod(get_poisson_loss)
    measure_dataset = "test"
    measure_attribute = "test_poissonloss"


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


@schema
class R2erEnsemble(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_r2er)
    measure_dataset = "test"
    measure_attribute = "r2_er"
    data_cache = None
    model_cache = None


@schema
class TestCorrEnsembleBlueSet(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test"
    measure_attribute = "test_correlation_blue"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="imagenet_v2_rgb_blue_only")


@schema
class TestCorrEnsembleGreenSet(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test"
    measure_attribute = "test_correlation_green"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="imagenet_v2_rgb_green_only")


@schema
class TestCorrEnsembleDepSet(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test"
    measure_attribute = "test_correlation_dependent"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(
        image_condition="imagenet_v2_rgb_gb_g_biased_correlated"
    )


@schema
class TestCorrEnsembleDepSetHighMSE(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test"
    measure_attribute = "test_correlation_dependent"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="imagenet_v2_rgb")


@schema
class CorrToAvgEnsembleBlueSet(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_correlations)
    measure_dataset = "test"
    measure_attribute = "avg_test_correlation_blue"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="imagenet_v2_rgb_blue_only")


@schema
class CorrToAvgEnsembleGreenSet(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_correlations)
    measure_dataset = "test"
    measure_attribute = "avg_test_correlation_green"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="imagenet_v2_rgb_green_only")


@schema
class CorrToAvgEnsembleDepSet(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_correlations)
    measure_dataset = "test"
    measure_attribute = "avg_test_correlation_dependent"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(
        image_condition="imagenet_v2_rgb_gb_g_biased_correlated"
    )


@schema
class CorrToAvgEnsembleDepSetHighMSE(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_correlations)
    measure_dataset = "test"
    measure_attribute = "avg_test_correlation_dependent"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="imagenet_v2_rgb")


@schema
class CtAEnsembleBlueHigh(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_correlations)
    measure_dataset = "test"
    measure_attribute = "avg_test_corr_blue_high"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="image_class_rgb_blue_high")


@schema
class CtAEnsembleBlueHigh(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_correlations)
    measure_dataset = "test"
    measure_attribute = "avg_test_corr_blue_high"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="image_class_rgb_blue_high")


@schema
class CtAEnsembleBlueLow(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_correlations)
    measure_dataset = "test"
    measure_attribute = "avg_test_corr_blue_low"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="imagenet_v2_blue_only_bckgr")


@schema
class CtAEnsembleGreenHigh(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_correlations)
    measure_dataset = "test"
    measure_attribute = "avg_test_corr_green_high"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="imagenet_v2_rgb_green_high")


@schema
class CtAEnsembleGreenLow(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_correlations)
    measure_dataset = "test"
    measure_attribute = "avg_test_corr_green_low"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="imagenet_v2_green_only_bckgr")


#### Center Surround Test Sets

@schema
class CtAeCSCenter(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_correlations)
    measure_dataset = "test"
    measure_attribute = "ctae_cs_center"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="imagenet_color_center")

@schema
class CtAeCSSurr(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_correlations)
    measure_dataset = "test"
    measure_attribute = "ctae_cs_surr"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="imagenet_color_surround")

@schema
class CtAeCSCenterSurrMixed(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_correlations)
    measure_dataset = "test"
    measure_attribute = "ctae_cs_center_surr_mixed"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="imagenet_color_center_surround_mixed")

@schema
class CtAeCSCenterSurrMixedSel(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_correlations)
    measure_dataset = "train"
    measure_attribute = "ctae_cs_center_surr_mixed"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="imagenet_color_center_surround_mixed_selected")

@schema
class CtAeCSFull(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_correlations)
    measure_dataset = "test"
    measure_attribute = "ctae_cs_full"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="imagenet_color_full")

@schema
class CtAeCSOpponentCenterGreen(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_correlations)
    measure_dataset = "test"
    measure_attribute = "ctae_cs_opp_center_green"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="imagenet_color_opponent_green_center")

@schema
class CtAeCSOpponentCenterUV(ScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_avg_correlations)
    measure_dataset = "test"
    measure_attribute = "ctae_cs_opp_center_uv"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="imagenet_color_opponent_uv_center")



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
    dataset_table = Dataset
    measure_function = staticmethod(get_fraction_oracles)
    measure_dataset = "test"
    measure_attribute = "fraction_oracle_jackknife"
    data_cache = None
    model_cache = None


@schema
class FractionOracleJackknifeEnsembleDepSetHighMSE(SummaryScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_fraction_oracles)
    measure_dataset = "test"
    measure_attribute = "fraction_oracle_jackknife_dependent"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="imagenet_v2_rgb")


@schema
class FractionOracleJackknifeEnsembleBlueSet(SummaryScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_fraction_oracles)
    measure_dataset = "test"
    measure_attribute = "fraction_oracle_jackknife_blue"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="imagenet_v2_rgb_blue_only")


@schema
class FractionOracleJackknifeEnsembleGreenSet(SummaryScoringTable):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MEISelector
    measure_function = staticmethod(get_fraction_oracles)
    measure_dataset = "test"
    measure_attribute = "fraction_oracle_jackknife_green"
    data_cache = DataCache
    model_cache = EnsembleModelCache
    dataloader_function_kwargs = dict(image_condition="imagenet_v2_rgb_green_only")



@schema
class RecMichelsonContrast(MEIScore):
    mei_table = Reconstruction
    measure_function = staticmethod(get_mei_michelson_contrast)
    measure_attribute = "michelson_contrast"
    external_download_path = fetch_download_path


@schema
class MEINorm(MEIScore):
    measure_function = staticmethod(get_mei_norm)
    measure_attribute = "mei_norm"
    external_download_path = fetch_download_path


@schema
class MEINormBlue(MEIScore):
    measure_function = staticmethod(get_mei_norm)
    measure_attribute = "mei_norm"
    external_download_path = fetch_download_path
    function_kwargs = dict(channel=1)


@schema
class MEINormGreen(MEIScore):
    measure_function = staticmethod(get_mei_norm)
    measure_attribute = "mei_norm"
    external_download_path = fetch_download_path
    function_kwargs = dict(channel=0)


@schema
class MEIColorBias(MEIScore):
    measure_function = staticmethod(get_mei_color_bias)
    measure_attribute = "mei_color_bias"
    external_download_path = fetch_download_path