from torch.utils.data import DataLoader
import torch
import torch.utils.data as utils
import numpy as np
#from retina.retina import warp_image
from skimage.transform import rescale
from collections import namedtuple, Iterable
import os
from neuralpredictors.data.samplers import RepeatsBatchSampler
import datajoint as dj

from dataport.bcm.static import PreprocessedMouseData
from neuralpredictors.data.datasets import FileTreeDataset
from pathlib import Path
import h5py
from tqdm import tqdm


def get_oracle_dataloader(dat,
                          toy_data=False,
                          image_condition=None,
                          verbose=False,
                          file_tree=False,
                          data_key=None):

    if toy_data:
        condition_hashes = dat.info.condition_hash
    else:
        dat_info = dat.info if not file_tree else dat.trial_info
        if 'image_id' in dir(dat_info):
            condition_hashes = dat_info.image_id
            image_class = dat_info.image_class

        elif 'colorframeprojector_image_id' in dir(dat_info):
            condition_hashes = dat_info.colorframeprojector_image_id
            image_class = dat_info.colorframeprojector_image_class
        elif 'frame_image_id' in dir(dat_info):
            condition_hashes = dat_info.frame_image_id
            image_class = dat_info.frame_image_class
        else:
            raise ValueError("'image_id' 'colorframeprojector_image_id', or 'frame_image_id' have to present in the dataset under dat.info "
                             "in order to load get the oracle repeats.")

    max_idx = condition_hashes.max() + 1
    classes, class_idx = np.unique(image_class, return_inverse=True)
    identifiers = condition_hashes + class_idx * max_idx
    dat_tiers = dat.tiers if not file_tree else dat.trial_info.tiers

    if image_condition is None:
        sampling_condition = np.where(dat_tiers == 'test')[0]
    elif isinstance(image_condition, str):
        image_condition_filter = image_class == image_condition
        sampling_condition = np.where((dat_tiers == 'test') & (image_condition_filter))[0]
    elif isinstance(image_condition, list):
        image_condition_filter = sum([image_class == i for i in image_condition]).astype(np.bool)
        sampling_condition = np.where((dat_tiers == 'test') & (image_condition_filter))[0]
    else:
        raise TypeError("image_condition argument has to be a string or list of strings")

    if (image_condition is not None) and verbose:
        print(f"Created Testloader for image class {image_condition}")

    sampler = RepeatsBatchSampler(identifiers, sampling_condition)
    dataloaders = {}
    dataloaders[data_key] = DataLoader(dat, batch_sampler=sampler)
    return dataloaders


def get_validation_split(n_images, train_frac, seed):
    """
    Splits the total number of images into train and test set.
    This ensures that in every session, the same train and validation images are being used.

    Args:
        n_images: Total number of images. These will be plit into train and validation set
        train_frac: fraction of images used for the training set
        seed: random seed

    Returns: Two arrays, containing image IDs of the whole imageset, split into train and validation

    """
    if seed: np.random.seed(seed)
    train_idx, val_idx = np.split(np.random.permutation(int(n_images)), [int(n_images*train_frac)])
    assert not np.any(np.isin(train_idx, val_idx)), "train_set and val_set are overlapping sets"

    return train_idx, val_idx


class ImageCache:
    """
    A simple cache which loads images into memory given a path to the directory where the images are stored.
    Images need to be present as 2D .npy arrays
    """

    def __init__(self, path=None, subsample=1, crop=0, scale=1, img_mean=None, img_std=None, transform=True, normalize=True, filename_precision=6):
        """

        path: str - pointing to the directory, where the individual .npy files are present
        subsample: int - amount of downsampling
        crop:  the expected input is a list of tuples, the specify the exact cropping from all four sides
                i.e. [(crop_left, crop_right), (crop_top, crop_down)]
        scale: - the scale factor to upsample or downsample images via interpolation
        img_mean: - mean luminance across all images
        img_std: - std of the luminance across all images
        transform: - whether to apply a transformation to an image
        normalize: - whether to standarized inputs by the mean and variance
        filename_precision: - amount leading zeros of the files in the specified folder
        """
        
        self.cache = {}
        self.path = path
        self.subsample = subsample
        self.crop = crop
        self.scale = scale
        self.img_mean = img_mean
        self.img_std = img_std
        self.transform = transform
        self.normalize = normalize
        self.leading_zeros = filename_precision

    def __len__(self):
        return len([file for file in os.listdir(self.path) if file.endswith('.npy')])

    def __contains__(self, key):
        return key in self.cache

    def __getitem__(self, item):
        item = item.tolist() if isinstance(item, Iterable) else item
        return [self[i] for i in item] if isinstance(item, Iterable) else self.update(item)

    def update(self, key):
        if key in self.cache:
            return self.cache[key]
        else:
            filename = os.path.join(self.path, str(key).zfill(self.leading_zeros) + '.npy')
            image = np.load(filename)
            image = self.transform_image(image) if self.transform else image
            image = self.normalize_image(image) if self.normalize else image
            image = torch.tensor(image).to(torch.float)
            self.cache[key] = image
            return image

    def transform_image(self, image):
        """
        applies transformations to the image: downsampling, cropping, rescaling, and dimension expansion.
        """
        
        h, w = image.shape
        rescale_fn = lambda x, s: rescale(x, s, mode='reflect', multichannel=False, anti_aliasing=False, preserve_range=True).astype(x.dtype) 
        image = image[self.crop[0][0]:h - self.crop[0][1]:self.subsample, self.crop[1][0]:w - self.crop[1][1]:self.subsample]
        image = image if self.scale == 1 else rescale_fn(image, self.scale)
        image = image[None,]
        return image
    
    def normalize_image(self, image):
        """
        standarizes image
        """
        image = (image - self.img_mean) / self.img_std
        return image
               

    @property
    def cache_size(self):
        return len(self.cache)
    
    @property
    def loaded_images(self):
        print('Loading images ...')
        items = [int(file.split('.')[0]) for file in os.listdir(self.path) if file.endswith('.npy')]
        images = torch.stack([self.update(item) for item in items])
        return images
    
    
    def zscore_images(self, update_stats=True):
        """
        zscore images in cache
        """
        images   = self.loaded_images
        img_mean = images.mean()
        img_std  = images.std()
        
        for key in self.cache:
            self.cache[key] = (self.cache[key] - img_mean) / img_std
        
        if update_stats:
            self.img_mean = np.float32(img_mean.item())
            self.img_std  = np.float32(img_std.item())
        


class CachedTensorDataset(utils.Dataset):
    """
    Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, names=('inputs', 'targets'), image_cache=None):
        if not all(tensors[0].size(0) == tensor.size(0) for tensor in tensors):
            raise ValueError('The tensors of the dataset have unequal lenghts. The first dim of all tensors has to match exactly.')
        if not len(tensors) == len(names):
            raise ValueError('Number of tensors and names provided have to match.  If there are more than two tensors,'
                             'names have to be passed to the TensorDataset')
        self.tensors = tensors
        self.input_position = names.index("inputs")
        self.DataPoint = namedtuple('DataPoint', names)
        self.image_cache = image_cache

    def __getitem__(self, index):
        """
        retrieves the inputs (= tensors[0]) from the image cache. If the image ID is not present in the cache,
            the cache is updated to load the corresponding image into memory.
        """
        if type(index) == int:
            key = self.tensors[0][index].item()
        else:
            key = self.tensors[0][index].numpy().astype(np.int32)

        tensors_expanded = [tensor[index] if pos != self.input_position else torch.stack(list(self.image_cache[key]))
                            for pos, tensor in enumerate(self.tensors)]

        return self.DataPoint(*tensors_expanded)

    def __len__(self):
        return self.tensors[0].size(0)


def get_cached_loader(image_ids, responses, batch_size, shuffle=True, image_cache=None, repeat_condition=None):
    """

    Args:
        image_ids: an array of image IDs
        responses: Numpy Array, Dimensions: N_images x Neurons
        batch_size: int - batch size for the dataloader
        shuffle: Boolean, shuffles image in the dataloader if True
        image_cache: a cache object which stores the images

    Returns: a PyTorch DataLoader object
    """

    image_ids = torch.tensor(image_ids.astype(np.int32))
    responses = torch.tensor(responses).to(torch.float)
    dataset = CachedTensorDataset(image_ids, responses, image_cache=image_cache)
    sampler = RepeatsBatchSampler(repeat_condition) if repeat_condition is not None else None

    dataloader = utils.DataLoader(dataset, batch_sampler=sampler) if batch_size is None else utils.DataLoader(dataset,
                                                                                                            batch_size=batch_size,
                                                                                                            shuffle=shuffle,
                                                                                                            )
    return dataloader


def add_h5_to_preprocessed_table(path, keys, comments, ignore_all_behaviors=True, filenames=None):
    """
    Args:
        path (str):     location of the h5 file to be added to the PreprocessedMouseData table.
        keys (list):    list of keys with the format dict(animal_id=.., session=.., scan_idx=..)
        comments (list): list of same length as keys, specifying the comment to be inserted into the
                            PreprocessedMouseData table

    Returns:
        filenames (list): filename of the dataset as a zip file. Corresponds to the attribute "filename"
                            in the PreprocessedMouseData table.
    """

    experiment = dj.create_virtual_module('experiment', 'sinzlab_houston_data')
    filename_template = 'static{animal_id}-{session}-{scan_idx}-preproc0'
    template = os.path.join(path, filename_template)
    if filenames is None:
        datasets = [(template + '.h5').format(**k)
                    for k in (experiment.Scan() & keys).fetch('KEY')]
    else:
        datasets = filenames

    print(datasets)
    for datafile in datasets:
        print(datafile)
        if datafile.endswith('.h5'):
            with h5py.File(datafile) as fid:
                print(datafile, fid['images'].shape)

    for datafile in datasets:
        if datafile.endswith('.h5'):
            FileTreeDataset.initialize_from(datafile, ignore_all_behaviors=ignore_all_behaviors)

    for key in (experiment.Scan() & keys).fetch('KEY'):
        if filenames is None:
            filename = (template + '/').format(**key)
        else:
            filename = datasets[0].split('.')[0]+'/'
        print(filename)
        dat = FileTreeDataset(filename, 'images', 'responses')
        dat.add_link('responses', 'targets')
        dat.add_link('images', 'inputs')
        print(dat)

    for key in (experiment.Scan() & keys).fetch('KEY'):
        print(key)
        if filenames is None:
            filename = (template + '/').format(**key)
        else:
            filename = datasets[0].split('.')[0] + '/'
        print(filename)
        dat = FileTreeDataset(filename, 'images', 'responses')
        ai, se, si, ui, x, y, z = (experiment.ScanSet.UnitInfo & key).fetch('animal_id', 'session', 'scan_idx',
                                                                            'unit_id', 'um_x', 'um_y', 'um_z')
        p = np.c_[x, y, z]
        dat.add_neuron_meta('cell_motor_coordinates', ai, se, si, ui, p)

    for key in (experiment.Scan() & keys).proj():
        if filenames is None:
            filename = (template + '/').format(**key)
        else:
            filename = datasets[0].split('.')[0] + '/'
        print(filename)
        dat = FileTreeDataset(filename, 'images', 'responses')
        dat.zip()

    if filenames is not None:
        PreprocessedMouseData().fill(filenames[0], comment=comments[0])
    else:
        filenames = []
        for i, key in enumerate((experiment.Scan() & keys).proj()):
            filename = Path((template + '.zip').format(**key))
            PreprocessedMouseData().fill(filename.name, comment=comments[i])
            filenames.append(filename.name)

    return filenames
