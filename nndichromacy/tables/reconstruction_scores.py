import datajoint as dj
import torch

from nnfabrik.utility.dj_helpers import CustomSchema
from .from_mei import MEIScore, ThresholdMEIMaskConfig
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
from .from_reconstruction import Reconstruction

schema = CustomSchema(dj.config.get("nnfabrik.schema_name", "nnfabrik_core"))
fetch_download_path = "/data/fetched_from_attach/"

@schema
class RecMichelsonContrast(MEIScore):
    mei_table = Reconstruction
    measure_function = staticmethod(get_mei_michelson_contrast)
    measure_attribute = "michelson_contrast"
    external_download_path = fetch_download_path

