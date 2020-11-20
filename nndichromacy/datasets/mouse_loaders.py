from collections import OrderedDict
from itertools import zip_longest
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from nnfabrik.utility.nn_helpers import set_random_seed
from neuralpredictors.data.datasets import StaticImageSet, FileTreeDataset
try:
    from neuralpredictors.data.transforms import Subsample, ToTensor, NeuroNormalizer, AddBehaviorAsChannels, SelectInputChannel, ScaleInputs, AddPupilCenterAsChannels
except:
    from neuralpredictors.data.transforms import Subsample, ToTensor, NeuroNormalizer, AddBehaviorAsChannels, SelectInputChannel


from neuralpredictors.data.samplers import SubsetSequentialSampler
from .utility import get_oracle_dataloader
from dataport.bcm.static import fetch_non_existing_data


@fetch_non_existing_data
def static_loader(
    path: str=None,
    batch_size: int=None,
    areas: list=None,
    layers: list=None,
    tier: str=None,
    neuron_ids: list=None,
    neuron_n: int=None,
    exclude_neuron_n=0,
    neuron_base_seed=None,
    image_ids=None,
    image_n=None,
    image_base_seed=None,
    get_key: bool=False,
    cuda: bool=True,
    normalize: bool=True,
    exclude: str=None,
    include_behavior: bool=False,
    select_input_channel: int=None,
    file_tree: bool=True,
    image_condition=None,
    return_test_sampler: bool=False,
    inputs_mean=None,
    inputs_std=None,
    scale: float=None,
    include_eye_position=None,
):
    """
    returns a single data loader

    Args:
        path (str): path for the dataset
        batch_size (int): batch size.
        areas (list, optional): the visual area.
        layers (list, optional): the layer from visual area.
        tier (str, optional): tier is a placeholder to specify which set of images to pick for train, val, and test loader.
        neuron_ids (list, optional): select neurons by their ids. neuron_ids and path should be of same length.
        neuron_n (int, optional): number of neurons to select randomly. Can not be set together with neuron_ids
        neuron_base_seed (float, optional): base seed for neuron selection. Get's multiplied by neuron_n to obtain final seed
        image_ids (list, optional): select images by their ids. image_ids and path should be of same length.
        image_n (int, optional): number of images to select randomly. Can not be set together with image_ids
        image_base_seed (float, optional): base seed for image selection. Get's multiplied by image_n to obtain final seed
        get_key (bool, optional): whether to return the data key, along with the dataloaders.
        cuda (bool, optional): whether to place the data on gpu or not.
        normalize (bool, optional): whether to normalize the data (see also exclude)
        exclude (str, optional): data to exclude from data-normalization. Only relevant if normalize=True. Defaults to 'images'
        include_behavior (bool, optional): whether to include behavioral data
        select_input_channel (int, optional): Only for color images. Select a color channel
        file_tree (bool, optional): whether to use the file tree dataset format. If False, equivalent to the HDF5 format
        image_condition (str, or list of str, optional): selection of images based on the image condition
        return_test_sampler (bool, optional): whether to return only the test loader with repeat-batches

    Returns:
        if get_key is False returns a dictionary of dataloaders for one dataset, where the keys are 'train', 'validation', and 'test'.
        if get_key is True it returns the data_key (as the first output) followed by the dataloder dictionary.

    """

    assert any([image_ids is None, all([image_n is None, image_base_seed is None])]), \
        "image_ids can not be set at the same time with anhy other image selection criteria"
    assert any([neuron_ids is None, all([neuron_n is None, neuron_base_seed is None, areas is None, layers is None, exclude_neuron_n==0])]), \
        "neuron_ids can not be set at the same time with any other neuron selection criteria"
    assert any([exclude_neuron_n==0, neuron_base_seed is not None]),  \
        "neuron_base_seed must be set when exclude_neuron_n is not 0"

    if image_ids is not None and image_condition is not None:
        raise ValueError("either 'image_condition' or 'image_ids' can be passed. They can not both be true.")

    data_keys = ["images", "responses",]
    if include_behavior:
        data_keys.append("behavior")
    if include_eye_position:
        data_keys.append("pupil_center")

    if file_tree:
        dat = FileTreeDataset(path, *data_keys)
    else:
        dat = StaticImageSet(path, *data_keys)

    assert (
        include_behavior and select_input_channel
    ) is not True, "Selecting an Input Channel and Adding Behavior can not both be true"

    # The permutation MUST be added first and the conditions below MUST NOT be based on the original order
    # specify condition(s) for sampling neurons. If you want to sample specific neurons define conditions that would effect idx
    conds = np.ones(len(dat.neurons.area), dtype=bool)
    if areas is not None:
        conds &= np.isin(dat.neurons.area, areas)
    if layers is not None:
        conds &= np.isin(dat.neurons.layer, layers)
    idx = np.where(conds)[0]
    if neuron_n is not None:
        random_state = np.random.get_state()
        if neuron_base_seed is not None:
            np.random.seed(neuron_base_seed * neuron_n) # avoid nesting by making seed dependent on number of neurons
        assert len(dat.neurons.unit_ids) >= exclude_neuron_n + neuron_n, \
            "After excluding {} neurons, there are not {} neurons left".format(exclude_neuron_n, neuron_n)
        neuron_ids = np.random.choice(dat.neurons.unit_ids, size=exclude_neuron_n + neuron_n, replace=False)[exclude_neuron_n:]
        np.random.set_state(random_state)
    if neuron_ids is not None:
        idx = [np.where(dat.neurons.unit_ids == unit_id)[0][0] for unit_id in neuron_ids]

    more_transforms = [Subsample(idx), ToTensor(cuda)]

    if select_input_channel is not None:
        more_transforms.insert(0, SelectInputChannel(select_input_channel))

    if include_eye_position:
        more_transforms.insert(0, AddPupilCenterAsChannels())

    if include_behavior:
        more_transforms.insert(0, AddBehaviorAsChannels())

    if normalize:
        try:
            more_transforms.insert(0, NeuroNormalizer(dat, exclude=exclude, inputs_mean=inputs_mean, inputs_std=inputs_std))
        except:
            more_transforms.insert(0, NeuroNormalizer(dat, exclude=exclude))

    if scale is not None:
        more_transforms.insert(0, ScaleInputs(scale=scale))

    dat.transforms.extend(more_transforms)

    # create the data_key for a specific data path
    data_key = path.split("static")[-1].split(".")[0].replace("preproc", "").replace("_nobehavior", "")

    if return_test_sampler:
        dataloader = get_oracle_dataloader(
            dat, image_condition=image_condition, file_tree=file_tree, data_key=data_key
        )
        return dataloader

    # subsample images
    dataloaders = {}
    keys = [tier] if tier else ["train", "validation", "test"]
    tier_array = dat.trial_info.tiers if file_tree else dat.tiers

    dat_info = dat.info if not file_tree else dat.trial_info
    if 'image_id' in dir(dat_info):
        frame_image_id = dat_info.image_id
        image_class = dat_info.image_class
    elif 'colorframeprojector_image_id' in dir(dat_info):
        frame_image_id = dat_info.colorframeprojector_image_id
        image_class = dat_info.colorframeprojector_image_class
    elif 'frame_image_id' in dir(dat_info):
        frame_image_id = dat_info.frame_image_id
        image_class = dat_info.frame_image_class
    else:
        raise ValueError(
            "'image_id' 'colorframeprojector_image_id', or 'frame_image_id' have to present in the dataset under dat.info "
            "in order to load get the oracle repeats.")

    if isinstance(image_condition, str):
        image_condition_filter = image_class == image_condition
    elif isinstance(image_condition, list):
        image_condition_filter = sum([image_class == i for i in image_condition]).astype(np.bool)
    else:
        if image_condition is not None:
            raise TypeError("image_condition argument has to be a string or list of strings")


    image_id_array = frame_image_id
    for tier in keys:
        # sample images
        if tier == "train" and image_ids is not None and image_condition is None:
            subset_idx = [np.where(image_id_array == image_id)[0][0] for image_id in image_ids]
            assert sum(tier_array[subset_idx] != "train") == 0, "image_ids contain validation or test images"
        elif tier == "train" and image_n is not None and image_condition is None:
            random_state = np.random.get_state()
            if image_base_seed is not None:
                np.random.seed(image_base_seed * image_n) #avoid nesting by making seed dependent on number of images
            subset_idx = np.random.choice(np.where(tier_array == 'train')[0], size=image_n, replace=False)
            np.random.set_state(random_state)
        elif image_condition is not None and image_ids is None:
            subset_idx = np.where(np.logical_and(image_condition_filter, tier_array == tier))[0]
            assert sum(tier_array[subset_idx] != tier) == 0, "image_ids contain validation or test images"
        else:
            subset_idx = np.where(tier_array == tier)[0]

        sampler = SubsetRandomSampler(subset_idx) if tier == "train" else SubsetSequentialSampler(subset_idx)
        dataloaders[tier] = DataLoader(dat, sampler=sampler, batch_size=batch_size)

    return (data_key, dataloaders) if get_key else dataloaders


def static_loaders(
    paths,
    batch_size: int,
    seed: int=None,
    areas: list=None,
    layers: list=None,
    tier: str=None,
    neuron_ids: list=None,
    neuron_n: int=None,
    exclude_neuron_n=0,
    neuron_base_seed=None,
    image_ids=None,
    image_n=None,
    image_base_seed=None,
    cuda: bool=True,
    normalize: bool=True,
    include_behavior: bool=False,
    exclude: str=None,
    select_input_channel: int=None,
    file_tree: bool=True,
    image_condition=None,
    inputs_mean=None,
    inputs_std=None,
    scale: float=None,
    include_eye_position=None,
):
    """
    Returns a dictionary of dataloaders (i.e., trainloaders, valloaders, and testloaders) for >= 1 dataset(s).

    Args:
        paths (list): list of paths for the datasets
        batch_size (int): batch size.
        seed (int): seed. Not really needed because there are neuron and image seed. But nnFabrik requires it.
        areas (list, optional): the visual area.
        layers (list, optional): the layer from visual area.
        tier (str, optional): tier is a placeholder to specify which set of images to pick for train, val, and test loader.
        neuron_ids (list, optional): select neurons by their ids. neuron_ids and path should be of same length.
        neuron_n (int, optional): number of neurons to select randomly. Can not be set together with neuron_ids
        exclude_neuron_n (int): the first <exclude_neuron_n> neurons will be excluded (given a neuron_base_seed),
                                then <neuron_n> neurons will be drawn from the remaining neurons.
        neuron_base_seed (float, optional): base seed for neuron selection. Get's multiplied by neuron_n to obtain final seed
        image_ids (list, optional): select images by their ids. image_ids and path should be of same length.
        image_n (int, optional): number of images to select randomly. Can not be set together with image_ids
        image_base_seed (float, optional): base seed for image selection. Get's multiplied by image_n to obtain final seed
        cuda (bool, optional): whether to place the data on gpu or not.
        normalize (bool, optional): whether to normalize the data (see also exclude)
        exclude (str, optional): data to exclude from data-normalization. Only relevant if normalize=True. Defaults to 'images'
        include_behavior (bool, optional): whether to include behavioral data
        select_input_channel (int, optional): Only for color images. Select a color channel
        file_tree (bool, optional): whether to use the file tree dataset format. If False, equivalent to the HDF5 format

    Returns:
        dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
    """
    if seed is not None:
        set_random_seed(seed)
    dls = OrderedDict({})
    keys = [tier] if tier else ["train", "validation", "test"]
    for key in keys:
        dls[key] = OrderedDict({})

    neuron_ids = [neuron_ids] if neuron_ids is None else neuron_ids
    image_ids = [image_ids] if image_ids is None else image_ids
    for path, neuron_id, image_id in zip_longest(paths, neuron_ids, image_ids, fillvalue=None):
        data_key, loaders = static_loader(
            path,
            batch_size,
            areas=areas,
            layers=layers,
            cuda=cuda,
            tier=tier,
            get_key=True,
            neuron_ids=neuron_id,
            neuron_n=neuron_n,
            exclude_neuron_n=exclude_neuron_n,
            neuron_base_seed=neuron_base_seed,
            image_ids=image_id,
            image_n=image_n,
            image_base_seed=image_base_seed,
            normalize=normalize,
            include_behavior=include_behavior,
            exclude=exclude,
            select_input_channel=select_input_channel,
            file_tree=file_tree,
            image_condition=image_condition,
            inputs_mean=inputs_mean,
            inputs_std=inputs_std,
            scale=scale,
            include_eye_position=include_eye_position,
        )
        for k in dls:
            dls[k][data_key] = loaders[k]

    return dls


def static_shared_loaders(
    paths,
    batch_size,
    seed=None,
    areas=None,
    layers=None,
    tier=None,
    multi_match_ids=None,
    multi_match_n=None,
    exclude_multi_match_n=0,
    multi_match_base_seed=None,
    image_ids=None,
    image_n=None,
    image_base_seed=None,
    cuda=True,
    normalize=True,
    include_behavior=False,
    exclude="images",
    select_input_channel=None,
):
    """
    Returns a dictionary of dataloaders (i.e., trainloaders, valloaders, and testloaders) for >= 1 dataset(s).
    The datasets must have matched neurons. Only the file tree format is supported.

    Args:
        paths (list): list of paths for the datasets
        batch_size (int): batch size.
        seed (int): seed. Not really needed because there are neuron and image seed. But nnFabrik requires it.
        areas (list, optional): the visual area.
        layers (list, optional): the layer from visual area.
        tier (str, optional): tier is a placeholder to specify which set of images to pick for train, val, and test loader.
        multi_match_ids (list, optional): select neurons by their ids. neuron_ids and path should be of same length.
        multi_match_n (int, optional): number of neurons to select randomly. Can not be set together with multi_match_ids
        exclude_multi_match_n (int): the first <exclude_multi_match_n> matched neurons will be excluded (given a multi_match_base_seed),
                                then <multi_match_n> matched neurons will be drawn from the remaining neurons.
        multi_match_base_seed (float, optional): base seed for neuron selection. Get's multiplied by multi_match_n to obtain final seed
        image_ids (list, optional): select images by their ids. image_ids and path should be of same length.
        image_n (int, optional): number of images to select randomly. Can not be set together with image_ids
        image_base_seed (float, optional): base seed for image selection. Get's multiplied by image_n to obtain final seed
        cuda (bool, optional): whether to place the data on gpu or not.
        normalize (bool, optional): whether to normalize the data (see also exclude)
        exclude (str, optional): data to exclude from data-normalization. Only relevant if normalize=True. Defaults to 'images'
        include_behavior (bool, optional): whether to include behavioral data
        select_input_channel (int, optional): Only for color images. Select a color channel

    Returns:
        dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
    """
    if seed is not None:
        set_random_seed(seed)

    assert (
        len(paths) != 1
    ), "Only one dataset was specified in 'paths'. When using the 'static_shared_loaders', more than one dataset has to be passed."
    assert any([multi_match_ids is None, all([multi_match_n is None, multi_match_base_seed is None, exclude_multi_match_n==0])]), \
        "multi_match_ids can not be set at the same time with any other multi_match selection criteria"
    assert any([exclude_multi_match_n==0, multi_match_base_seed is not None]),  \
        "multi_match_base_seed must be set when exclude_multi_match_n is not 0"

    # Collect overlapping multi matches
    multi_unit_ids, per_data_set_ids, given_neuron_ids = [], [], []
    match_set = None
    for path in paths:
        data_key, dataloaders = static_loader(path=path, batch_size=100, get_key=True)
        dat = dataloaders['train'].dataset
        multi_unit_ids.append(dat.neurons.multi_match_id)
        per_data_set_ids.append(dat.neurons.unit_ids)
        if match_set is None:
            match_set = set(multi_unit_ids[-1])
        else:
            match_set &= set(multi_unit_ids[-1])
        if multi_match_ids is not None:
            assert set(multi_match_ids).issubset(
                dat.neurons.multi_match_id
            ), "Dataset {} does not contain all multi_match_ids".format(path)
            neuron_idx = [
                np.where(dat.neurons.multi_match_id == multi_match_id)[0][0] for multi_match_id in multi_match_ids
            ]
            given_neuron_ids.append(dat.neurons.unit_ids[neuron_idx])
    match_set -= {-1}  # remove the unmatched neurons
    match_set = np.array(list(match_set))

    # get unit_ids of matched neurons
    if multi_match_ids is not None:
        neuron_ids = given_neuron_ids
    elif multi_match_n is not None:
        random_state = np.random.get_state()
        if multi_match_base_seed is not None:
            np.random.seed(multi_match_base_seed * multi_match_n) # avoid nesting by making seed dependent on number of neurons
        assert len(match_set) >= exclude_multi_match_n + multi_match_n, \
            "After excluding {} neurons, there are not {} matched neurons left".format(exclude_multi_match_n, multi_match_n)
        match_subset = np.random.choice(match_set, size=exclude_multi_match_n + multi_match_n, replace=False)[exclude_multi_match_n:]
        neuron_ids = [pdsi[np.isin(munit_ids, match_subset)] for munit_ids, pdsi in zip(multi_unit_ids, per_data_set_ids)]
        np.random.set_state(random_state)
    else:
        neuron_ids = [pdsi[np.isin(munit_ids, match_set)] for munit_ids, pdsi in zip(multi_unit_ids, per_data_set_ids)]

    # generate single dataloaders
    dls = OrderedDict({})
    keys = [tier] if tier else ["train", "validation", "test"]
    for key in keys:
        dls[key] = OrderedDict({})

    image_ids = [image_ids] if image_ids is None else image_ids
    for path, neuron_id, image_id in zip_longest(paths, neuron_ids, image_ids, fillvalue=None):
        data_key, loaders = static_loader(
            path,
            batch_size,
            areas=areas,
            layers=layers,
            cuda=cuda,
            tier=tier,
            get_key=True,
            neuron_ids=neuron_id,
            neuron_n=None,
            neuron_base_seed=None,
            image_ids=image_id,
            image_n=image_n,
            image_base_seed=image_base_seed,
            normalize=normalize,
            include_behavior=include_behavior,
            exclude=exclude,
            select_input_channel=select_input_channel,
            file_tree=True,
        )
        for k in dls:
            dls[k][data_key] = loaders[k]

    return dls
