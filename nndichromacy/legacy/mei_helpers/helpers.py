from featurevis.integration import load_pickled_data
import numpy as np
import types
import featurevis

def is_ensemble(model):
    return (isinstance(model, types.FunctionType) or isinstance(model, featurevis.integration.EnsembleModel))


def get_neuron_mappings(dataset_config, key, load_func=None):
    if load_func is None:
        load_func = load_pickled_data
    entities = []
    for i, datafile_path in enumerate(dataset_config["neuronal_data_files"]):
        data = load_func(datafile_path)
        unit_ids = data["unit_ids"] if "unit_ids" in data else np.arange(data["testing_responses"].shape[1])

        for neuron_pos, unit_id in enumerate(unit_ids):
            entities.append(dict(key, neuron_id=unit_id, neuron_position=neuron_pos, session_id=int(data["session_id"])))
    return entities


def get_real_mappings(dataset_config, key, load_func=None):
    if load_func is None:
        load_func = load_pickled_data
    entities = []
    for datafile_path in dataset_config["neuronal_data_files"]:
        data = load_func(datafile_path)
        for neuron_pos, neuron_id in enumerate(data["unit_indices"]):
            entities.append(dict(key, neuron_id=neuron_id, neuron_position=neuron_pos, session_id=data["session_id"]))
    return entities