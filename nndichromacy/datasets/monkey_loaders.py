import torch
import torch.utils.data as utils
import numpy as np
import pickle
#from retina.retina import warp_image
from collections import namedtuple, Iterable
import os
from mlutils.data.samplers import RepeatsBatchSampler
from .utility import get_validation_split, ImageCache, get_cached_loader
from nnfabrik.utility.nn_helpers import get_module_output, set_random_seed, get_dims_for_loader_dict
from nnfabrik.utility.dj_helpers import make_hash


def monkey_static_loader(dataset,
                         neuronal_data_files,
                         image_cache_path,
                         batch_size=64,
                         seed=None,
                         train_frac=0.8,
                         subsample=1,
                         crop=((96,96), (96,96)),
                         scale=1.,
                         time_bins_sum=tuple(range(12)),
                         avg=False,
                         image_file=None,
                         return_data_info=False,
                         store_data_info=True):
    """
    Function that returns cached dataloaders for monkey ephys experiments.

     creates a nested dictionary of dataloaders in the format
            {'train' : dict_of_loaders,
             'validation'   : dict_of_loaders,
            'test'  : dict_of_loaders, }

        in each dict_of_loaders, there will be  one dataloader per data-key (refers to a unique session ID)
        with the format:
            {'data-key1': torch.utils.data.DataLoader,
             'data-key2': torch.utils.data.DataLoader, ... }

    requires the types of input files:
        - the neuronal data files. A list of pickle files, with one file per session
        - the image file. The pickle file that contains all images.
        - individual image files, stored as numpy array, in a subfolder

    Args:
        dataset: a string, identifying the Dataset:
            'PlosCB19_V1', 'CSRF19_V1', 'CSRF19_V4'
            This string will be parsed by a datajoint table

        neuronal_data_files: a list paths that point to neuronal data pickle files
        image_file: a path that points to the image file
        image_cache_path: The path to the cached images
        batch_size: int - batch size of the dataloaders
        seed: int - random seed, to calculate the random split
        train_frac: ratio of train/validation images
        subsample: int - downsampling factor
        crop: int or tuple - crops x pixels from each side. Example: Input image of 100x100, crop=10 => Resulting img = 80x80.
            if crop is tuple, the expected input is a list of tuples, the specify the exact cropping from all four sides
                i.e. [(crop_left, crop_right), (crop_top, crop_bottom)]
        scale: float or integer - up-scale or down-scale via interpolation hte input images (default= 1)
        time_bins_sum: sums the responses over x time bins.
        avg: Boolean - Sums oder Averages the responses across bins.

    Returns: nested dictionary of dataloaders
    """

    dataset_config = locals()

    # initialize dataloaders as empty dict
    dataloaders = {'train': {}, 'validation': {}, 'test': {}}

    if not isinstance(time_bins_sum, Iterable):
        time_bins_sum = tuple(range(time_bins_sum))

    if isinstance(crop, int):
        crop = [(crop, crop), (crop, crop)]

    # clean up image path because of legacy folder structure
    image_cache_path = image_cache_path.split('individual')[0]

    # Load image statistics if present
    stats_filename = make_hash(dataset_config)
    stats_path = os.path.join(image_cache_path, 'statistics/', stats_filename)
    
    # Get mean and std
   
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as pkl:
            data_info = pickle.load(pkl)
        if return_data_info:
            return data_info
        img_mean = list(data_info.values())[0]["img_mean"]
        img_std = list(data_info.values())[0]["img_std"]
        
        # Initialize cache
        cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, img_mean= img_mean, img_std=img_std, transform=True, normalize=True)
    else:
        # Initialize cache with no normalization
        cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, transform=True, normalize=False) 
        
        # Compute mean and std of transformed images and zscore data (the cache wil be filled so first epoch will be fast)
        cache.zscore_images(update_stats=True)
        img_mean = cache.img_mean
        img_std  = cache.img_std
    
    
    n_images = len(cache)
    data_info = {}

    # set up parameters for the different dataset types
    if dataset == 'PlosCB19_V1':
        # for the "Amadeus V1" Dataset, recorded by Santiago Cadena, there was no specified test set.
        # instead, the last 20% of the dataset were classified as test set. To make sure that the test set
        # of this dataset will always stay identical, the `train_test_split` value is hardcoded here.
        train_test_split = 0.8
        image_id_offset = 1
    else:
        train_test_split = 1
        image_id_offset = 0

    all_train_ids, all_validation_ids = get_validation_split(n_images=n_images * train_test_split,
                                                             train_frac=train_frac,
                                                             seed=seed)
    
    # cycling through all datafiles to fill the dataloaders with an entry per session
    for i, datapath in enumerate(neuronal_data_files):

        with open(datapath, "rb") as pkl:
            raw_data = pickle.load(pkl)

        subject_ids = raw_data["subject_id"]
        data_key = str(raw_data["session_id"])
        responses_train = raw_data["training_responses"].astype(np.float32)
        responses_test = raw_data["testing_responses"].astype(np.float32)
        training_image_ids = raw_data["training_image_ids"] - image_id_offset
        testing_image_ids = raw_data["testing_image_ids"] - image_id_offset

        if dataset != 'PlosCB19_V1':
            responses_test = responses_test.transpose((2, 0, 1))
            responses_train = responses_train.transpose((2, 0, 1))

            if time_bins_sum is not None:  # then average over given time bins
                responses_train = (np.mean if avg else np.sum)(responses_train[:, :, time_bins_sum], axis=-1)
                responses_test = (np.mean if avg else np.sum)(responses_test[:, :, time_bins_sum], axis=-1)

        train_idx = np.isin(training_image_ids, all_train_ids)
        val_idx = np.isin(training_image_ids, all_validation_ids)

        responses_val = responses_train[val_idx]
        responses_train = responses_train[train_idx]

        validation_image_ids = training_image_ids[val_idx]
        training_image_ids = training_image_ids[train_idx]

        train_loader = get_cached_loader(training_image_ids, responses_train, batch_size=batch_size, image_cache=cache)
        val_loader = get_cached_loader(validation_image_ids, responses_val, batch_size=batch_size, image_cache=cache)
        test_loader = get_cached_loader(testing_image_ids,
                                        responses_test,
                                        batch_size=None,
                                        shuffle=None,
                                        image_cache=cache,
                                        repeat_condition=testing_image_ids)

        dataloaders["train"][data_key] = train_loader
        dataloaders["validation"][data_key] = val_loader
        dataloaders["test"][data_key] = test_loader


    if store_data_info and not os.path.exists(stats_path):

        in_name, out_name = next(iter(list(dataloaders["train"].values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders["train"])
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = {k: v[in_name][1] for k, v in session_shape_dict.items()}

        for data_key in session_shape_dict:
            data_info[data_key] = dict(input_dimensions=in_shapes_dict[data_key],
                                       input_channels=input_channels[data_key],
                                       output_dimension=n_neurons_dict[data_key],
                                       img_mean=img_mean,
                                       img_std=img_std)

        
        with open(stats_path, "wb") as pkl:
            pickle.dump(data_info, pkl)

    return dataloaders if not return_data_info else data_info


def monkey_mua_sua_loader(dataset,
                         neuronal_data_files,
                         mua_data_files,
                         image_cache_path,
                         batch_size=64,
                         seed=None,
                         train_frac=0.8,
                         subsample=1,
                         crop=((96, 96), (96, 96)),
                         scale=1.,
                         time_bins_sum=tuple(range(12)),
                         avg=False,
                         image_file=None,
                         return_data_info=False,
                         store_data_info=True):
    """
    Function that returns cached dataloaders for monkey ephys experiments.

     creates a nested dictionary of dataloaders in the format
            {'train' : dict_of_loaders,
             'validation'   : dict_of_loaders,
            'test'  : dict_of_loaders, }

        in each dict_of_loaders, there will be  one dataloader per data-key (refers to a unique session ID)
        with the format:
            {'data-key1': torch.utils.data.DataLoader,
             'data-key2': torch.utils.data.DataLoader, ... }

    requires the types of input files:
        - the neuronal data files. A list of pickle files, with one file per session
        - the image file. The pickle file that contains all images.
        - individual image files, stored as numpy array, in a subfolder

    Args:
        dataset: a string, identifying the Dataset:
            'PlosCB19_V1', 'CSRF19_V1', 'CSRF19_V4'
            This string will be parsed by a datajoint table

        neuronal_data_files: a list paths that point to neuronal data pickle files
        image_file: a path that points to the image file
        image_cache_path: The path to the cached images
        batch_size: int - batch size of the dataloaders
        seed: int - random seed, to calculate the random split
        train_frac: ratio of train/validation images
        subsample: int - downsampling factor
        crop: int or tuple - crops x pixels from each side. Example: Input image of 100x100, crop=10 => Resulting img = 80x80.
            if crop is tuple, the expected input is a list of tuples, the specify the exact cropping from all four sides
                i.e. [(crop_left, crop_right), (crop_top, crop_bottom)]
        scale: float or integer - up-scale or down-scale via interpolation hte input images (default= 1)
        time_bins_sum: sums the responses over x time bins.
        avg: Boolean - Sums oder Averages the responses across bins.

    Returns: nested dictionary of dataloaders
    """

    dataset_config = locals()

    # initialize dataloaders as empty dict
    dataloaders = {'train': {}, 'validation': {}, 'test': {}}

    if not isinstance(time_bins_sum, Iterable):
        time_bins_sum = tuple(range(time_bins_sum))

    if isinstance(crop, int):
        crop = [(crop, crop), (crop, crop)]

    # clean up image path because of legacy folder structure
    image_cache_path = image_cache_path.split('individual')[0]

    # Load image statistics if present
    stats_filename = make_hash(dataset_config)
    stats_path = os.path.join(image_cache_path, 'statistics/', stats_filename)

    # Get mean and std

    if os.path.exists(stats_path):
        with open(stats_path, "rb") as pkl:
            data_info = pickle.load(pkl)
        if return_data_info:
            return data_info
        img_mean = list(data_info.values())[0]["img_mean"]
        img_std = list(data_info.values())[0]["img_std"]

        # Initialize cache
        cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, img_mean=img_mean,
                           img_std=img_std, transform=True, normalize=True)
    else:
        # Initialize cache with no normalization
        cache = ImageCache(path=image_cache_path, subsample=subsample, crop=crop, scale=scale, transform=True,
                           normalize=False)

        # Compute mean and std of transformed images and zscore data (the cache wil be filled so first epoch will be fast)
        cache.zscore_images(update_stats=True)
        img_mean = cache.img_mean
        img_std = cache.img_std

    n_images = len(cache)
    data_info = {}

    # set up parameters for the different dataset types
    if dataset == 'PlosCB19_V1':
        # for the "Amadeus V1" Dataset, recorded by Santiago Cadena, there was no specified test set.
        # instead, the last 20% of the dataset were classified as test set. To make sure that the test set
        # of this dataset will always stay identical, the `train_test_split` value is hardcoded here.
        train_test_split = 0.8
        image_id_offset = 1
    else:
        train_test_split = 1
        image_id_offset = 0

    all_train_ids, all_validation_ids = get_validation_split(n_images=n_images * train_test_split,
                                                             train_frac=train_frac,
                                                             seed=seed)

    # cycling through all datafiles to fill the dataloaders with an entry per session
    for i, datapath in enumerate(neuronal_data_files):

        with open(datapath, "rb") as pkl:
            raw_data = pickle.load(pkl)

        subject_ids = raw_data["subject_id"]
        data_key = str(raw_data["session_id"])
        responses_train = raw_data["training_responses"].astype(np.float32)
        responses_test = raw_data["testing_responses"].astype(np.float32)
        training_image_ids = raw_data["training_image_ids"] - image_id_offset
        testing_image_ids = raw_data["testing_image_ids"] - image_id_offset

        for mua_data_path in mua_data_files:
            with open(mua_data_path, "rb") as mua_pkl:
                mua_data = pickle.load(mua_pkl)

            if str(mua_data["session_id"]) == data_key:
                mua_responses_train = mua_data["training_responses"].astype(np.float32)
                mua_responses_test = mua_data["testing_responses"].astype(np.float32)
                mua_training_image_ids = mua_data["training_image_ids"] - image_id_offset
                mua_testing_image_ids = mua_data["testing_image_ids"] - image_id_offset
                break

        if not str(mua_data["session_id"]) == data_key:
            print("session {} does not exist in MUA. Skipping MUA".format(data_key))
        else:
            if not np.array_equal(training_image_ids, mua_training_image_ids):
                raise ValueError("Training image IDs do not match between the spike sorted data and mua data")
            if not np.array_equal(testing_image_ids, mua_testing_image_ids):
                raise ValueError("Testing image IDs do not match between the spike sorted data and mua data")
            responses_train = np.concatenate([responses_train, mua_responses_train], axis=0)
            responses_test = np.concatenate([responses_test, mua_responses_test], axis=0)


        if dataset != 'PlosCB19_V1':
            responses_test = responses_test.transpose((2, 0, 1))
            responses_train = responses_train.transpose((2, 0, 1))

            if time_bins_sum is not None:  # then average over given time bins
                responses_train = (np.mean if avg else np.sum)(responses_train[:, :, time_bins_sum], axis=-1)
                responses_test = (np.mean if avg else np.sum)(responses_test[:, :, time_bins_sum], axis=-1)

        train_idx = np.isin(training_image_ids, all_train_ids)
        val_idx = np.isin(training_image_ids, all_validation_ids)

        responses_val = responses_train[val_idx]
        responses_train = responses_train[train_idx]

        validation_image_ids = training_image_ids[val_idx]
        training_image_ids = training_image_ids[train_idx]

        train_loader = get_cached_loader(training_image_ids, responses_train, batch_size=batch_size, image_cache=cache)
        val_loader = get_cached_loader(validation_image_ids, responses_val, batch_size=batch_size, image_cache=cache)
        test_loader = get_cached_loader(testing_image_ids,
                                        responses_test,
                                        batch_size=None,
                                        shuffle=None,
                                        image_cache=cache,
                                        repeat_condition=testing_image_ids)

        dataloaders["train"][data_key] = train_loader
        dataloaders["validation"][data_key] = val_loader
        dataloaders["test"][data_key] = test_loader

    if store_data_info and not os.path.exists(stats_path):

        in_name, out_name = next(iter(list(dataloaders["train"].values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders["train"])
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = {k: v[in_name][1] for k, v in session_shape_dict.items()}

        for data_key in session_shape_dict:
            data_info[data_key] = dict(input_dimensions=in_shapes_dict[data_key],
                                       input_channels=input_channels[data_key],
                                       output_dimension=n_neurons_dict[data_key],
                                       img_mean=img_mean,
                                       img_std=img_std)

        with open(stats_path, "wb") as pkl:
            pickle.dump(data_info, pkl)

    return dataloaders if not return_data_info else data_info