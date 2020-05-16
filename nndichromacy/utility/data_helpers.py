from mlutils.data.transforms import Subsample, ToTensor, NeuroNormalizer, AddBehaviorAsChannels, SelectInputChannel
import numpy as np
from torch.utils.data import DataLoader
from mlutils.data.datasets import StaticImageSet
from mlutils.data.samplers import RepeatsBatchSampler


def get_oracle_dataloader(paths=None, seed=None, area='V1', layer='L2/3',
                          tier=None, neuron_ids=None, get_key=False, cuda=True, normalize=True, include_behavior=False,
                          exclude=None, select_input_channel=None, toy_data=False, **kwargs):
    assert (
                       include_behavior and select_input_channel) is False, "Selecting an Input Channel and Adding Behavior can not both be true"

    path = paths[0]
    dat = StaticImageSet(path, 'images', 'responses', 'behavior') if include_behavior else StaticImageSet(path,
                                                                                                          'images',
                                                                                                          'responses')
    # add all transforms
    if toy_data:
        dat.transforms = [ToTensor(cuda)]
    else:
        # specify condition(s) for sampling neurons. If you want to sample specific neurons define conditions that would effect idx
        neuron_ids = neuron_ids if neuron_ids else dat.neurons.unit_ids
        conds = ((dat.neurons.area == area) &
                 (dat.neurons.layer == layer) &
                 (np.isin(dat.neurons.unit_ids, neuron_ids)))
        idx = np.where(conds)[0]
        dat.transforms = [Subsample(idx), ToTensor(cuda)]
        if normalize:
            dat.transforms.insert(0, NeuroNormalizer(dat, exclude=exclude))
        if include_behavior:
            dat.transforms.insert(0, AddBehaviorAsChannels())
        if select_input_channel is not None:
            dat.transforms.insert(0, SelectInputChannel(select_input_channel))

    # get repeated conditions
    types = np.unique(dat.types)
    if len(types) == 1 and types[0] == "stimulus.ColorFrameProjector":
        if 'image_id' in dir(dat.info):
            condition_hashes = dat.info.image_id
        elif 'colorframeprojector_image_id' in dir(dat.info):
            condition_hashes = dat.info.colorframeprojector_image_id
        else:
            condition_hashes = dat.info.condition_hash

    else:
        raise TypeError("Do not recognize types={}".format(*types))

    if toy_data:
        condition_hashes = dat.info.condition_hash
    dataloader = DataLoader(dat, batch_sampler=RepeatsBatchSampler(condition_hashes))
    return dataloader
