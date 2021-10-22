import datajoint as dj
from nnfabrik.utility.nnf_helper import FabrikCache
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from .from_mei import MEI, TrainedEnsembleModel, MEISeed, MEISelector, MEIMethod
from .from_nnfabrik import TrainedModel
from nnfabrik.utility.dj_helpers import CustomSchema

schema = CustomSchema(dj.config.get("nnfabrik.schema_name", "nnfabrik_core"))


DataCache = FabrikCache(base_table=Dataset, cache_size_limit=1)
TrainedModelCache = FabrikCache(base_table=TrainedModel, cache_size_limit=1)
EnsembleModelCache = FabrikCache(base_table=TrainedEnsembleModel, cache_size_limit=1)
