#!/usr/bin/python3

# DJ and utils imports
import datajoint as dj
import time
import tarfile
from shutil import copyfile
import os

dj.config["enable_python_native_blobs"] = True
dj.config["nnfabrik.schema_name"]= 'nnfabrik_color_mei'
schema = dj.schema('nnfabrik_color_mei')

# copy data from QB to $SCRATCH volume
os.makedirs('/data/mouse/toliaslab/static/')


# project specific imports
from nndichromacy.tables.from_nnfabrik import TrainedModel

dataset_hashes = ['f6b2477ef169c1e1f40b9d125cb2b520']
for dataset_hash in dataset_hashes:
    pop_key = {'dataset_hash': dataset_hash,
               'trainer_hash': '0d06f037501e129d11aa288d8f22788f',
               'model_hash': 'af90cf960a14d3ea4b8d6138d4510693'}
    TrainedModel.populate(pop_key, display_progress=True, reserve_jobs=True)