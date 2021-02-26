#!/usr/bin/python3

# DJ and utils imports
import datajoint as dj
import time
import tarfile
from shutil import copyfile
import os

dj.config["enable_python_native_blobs"] = True
dj.config["nnfabrik.schema_name"]= 'nnfabrik_toy_V4'
schema = dj.schema('nnfabrik_toy_V4')

# copy data from QB to $SCRATCH volume
os.makedirs('/data/mouse/toliaslab/static/')


# project specific imports
from nndichromacy.tables.from_nnfabrik import TrainedModel


print("Entries in TrainedModel table", len(TrainedModel()))