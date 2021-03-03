import datajoint as dj

dj.config["enable_python_native_blobs"] = True
dj.config['nnfabrik.schema_name'] = "nnfabrik_color_mei"
dj.config["display.limit"] = 70
schema = dj.schema("nnfabrik_color_mei")

from nndichromacy.tables.from_mei import MEI, MEIMethod, MEISeed, MEISelector, TrainedEnsembleModel

from nndichromacy.tables.from_nnfabrik import TrainedModel
from nndichromacy.tables.from_mei import MEI, MEIMethod, MEISeed, MEISelector, TrainedEnsembleModel



for dataset_hash in ['eaae92b3f2c085cf5b35d0faddc63c15']:
    for method_hash in ['4f899fe8aaadbebf734c46f167028614', '782dfe2d7c6c0be6d3d018419646fc4c']:

        mei_key = dict(method_hash=method_hash, dataset_hash=dataset_hash)
        #unit_keys = (SignalToNoiseRatio.Units & mei_key & "unit_snr > 0.3").fetch("KEY", limit=300)
        #pop_key = dj.AndList([unit_keys, mei_key])
        #MEI.populate(pop_key, display_progress=True, order='random', reserve_jobs=True)

        MEI.populate(mei_key, display_progress=True, order='random', reserve_jobs=True)