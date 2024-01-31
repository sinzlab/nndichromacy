#!/usr/bin/python3

import os

import datajoint as dj
dj.config["database.host"] = '134.76.19.44'
dj.config["enable_python_native_blobs"] = True
dj.config['nnfabrik.schema_name'] = "nnfabrik_color_mei"
schema = dj.schema("nnfabrik_color_mei")

from nnfabrik.main import *
from nndichromacy.tables.from_mei import MEI, MEIMethod, MEISeed, MEISelector, TrainedEnsembleModel
from nndichromacy.tables.from_nnfabrik import TrainedModel
from nndichromacy.tables.mei_scores import MEIMichelsonContrast
from nndichromacy.tables.from_reconstruction import ReconMethod, ReconObjective, Reconstruction, ReconTargetFunction, ReconTargetUnit, ReconType
from dataport.bcm.color_mei.schema import StaticImage

from nndichromacy.tables.scores import TestCorrelationEnsemble, CorrelationToAvergeEnsemble

method_keys = (MEIMethod & "method_ts >'2022-09-01'").fetch("KEY")

ids_10 = [990, 1149, 372, 405, 497, 788, 791, 931, 1031, 1132]
for unit_id in ids_10:
    mei_key = dj.AndList([method_keys, dict(unit_id=unit_id, ensemble_hash="4b4d819b947a5c190ca9ee1e39820afb", mei_seed=1)])
    MEI().populate(mei_key, display_progress=True, reserve_jobs=True)