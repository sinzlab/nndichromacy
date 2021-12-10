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

schema = CustomSchema(dj.config.get("nnfabrik.schema_name", "nnfabrik_core"))
fetch_download_path = "/data/fetched_from_attach/"


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


@schema
class MEIMichelsonContrast(MEIScore):
    measure_function = staticmethod(get_mei_michelson_contrast)
    measure_attribute = "michelson_contrast"
    external_download_path = fetch_download_path


@schema
class MEIThresholdMask(MEIScore):
    """
    A template for a MEI scoring table.
    """

    external_download_path = fetch_download_path

    # table level comment
    table_comment = "Calculates the default thresholded MEI masks, based on configs of the MaskConfig table"

    @property
    def definition(self):
        definition = """
                    # {table_comment}
                    -> self.mei_table
                    -> ThresholdMEIMaskConfig
                    ---
                    mask:      longblob     # A template for a computed score of a trained model
                    mask_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
                    """.format(table_comment=self.table_comment, )
        return definition

    def get_mei(self, key):
        mei = torch.load(
            (self.mei_table & key).fetch1(
                "mei", download_path=self.external_download_path
            )
        )
        return mei

    @staticmethod
    def measure_function(mei, **kwargs):
        raise NotImplementedError("Scoring Function has to be implemented")

    def make(self, key):
        mei = self.get_mei(key=key)
        mask_fn, mask_config = (ThresholdMEIMaskConfig & key).fn_config
        mask = mask_fn(mei, **mask_config)
        key["mask"] = mask
        self.insert1(key, ignore_extra_fields=True)


from nndichromacy.tables.surroundMEI import SurroundMEI
@schema
class MEISurroundMichelsonContrast(MEIScore):
    mei_table = SurroundMEI
    measure_function = staticmethod(get_mei_michelson_contrast)
    measure_attribute = "michelson_contrast"
    external_download_path = fetch_download_path


