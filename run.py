#!/usr/bin/python3
import datajoint as dj

dj.config["enable_python_native_blobs"] = True
dj.config['nnfabrik.schema_name'] = "nnfabrik_color_mei"
dj.config["display.limit"] = 70
schema = dj.schema("nnfabrik_color_mei")

from nndichromacy.tables.from_mei import MEI, MEIMethod, MEISeed

from nndichromacy.mei.initial import RandomNormalBehavior
print("import successful")

key = {'dataset_hash': 'a814c139b67e879a351442d71f62f3c2'}
from nndichromacy.tables.scores import TestCorrelationEnsemble, CorrelationToAvergeEnsemble
method_keys = (MEIMethod & "method_ts >'2021-03-31'").fetch("KEY")
unit_keys = (CorrelationToAvergeEnsemble.Units & key).fetch("KEY", limit=150, order_by="unit_avg_correlation DESC")

mei_key = dj.AndList([unit_keys, method_keys, dict(dataset_hash='a814c139b67e879a351442d71f62f3c2')])

MEI().populate(mei_key, display_progress=True, reserve_jobs=True)