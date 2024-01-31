from abc import ABC, abstractmethod
import torch
from torch import Tensor, randn
from nndichromacy.tables.from_mei import MEI
from nnfabrik import builder

import os

fetch_download_path = os.environ.get('FETCH_DOWNLOAD_PATH', '/data/fetched_from_attach')


class InitialGuessCreator(ABC):
    """Implements the interface used to create an initial guess."""

    @abstractmethod
    def __call__(self, *shape) -> Tensor:
        """Creates an initial guess from which to start the MEI optimization process given a shape."""


class RandomNormal(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        return self._create_random_tensor(*shape)


    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class RandomNormalNullChannel(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __init__(self, null_channel, null_value=0):
        self.null_channel = null_channel
        self.null_value = null_value

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        inital = self._create_random_tensor(*shape)
        inital[:, self.null_channel, ...] = self.null_value
        return inital

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"
    
class RandomNormalCenterRing(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __init__(self, key, mask_thres_for_ring=0.3):
        src_method_fn = key["src_method_fn"]
        unit_id = key["unit_id"]

        inner_ensemble_hash = key["inner_ensemble_hash"]
        outer_ensemble_hash = key["outer_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        outer_method_hash = key["outer_method_hash"]
        unit_id = key["unit_id"]

        outer_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=outer_ensemble_hash) & dict(method_hash=outer_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        inner_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=inner_ensemble_hash) & dict(method_hash=inner_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        
        outer_mei=torch.load(outer_mei_path)
        inner_mei=torch.load(inner_mei_path)

        self.centerimg = inner_mei[0][0]
        self.ring_mask=(outer_mei[0][1] - inner_mei[0][1] > mask_thres_for_ring) * 1

        
    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        randinitial = self._create_random_tensor(*shape)
        initial = randinitial * self.ring_mask + self.centerimg
        return initial

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"
    
class RandomNormalNonlinearCenterRing(InitialGuessCreator):

    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""
    ### put cropped nonlinear MEI in center
    _create_random_tensor = randn

    def __init__(self, key, mask_thres_for_ring=0.3):
        src_method_fn = key["src_method_fn"]
        unit_id = key["unit_id"]

        inner_ensemble_hash = key["inner_ensemble_hash"]
        outer_ensemble_hash = key["outer_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        outer_method_hash = key["outer_method_hash"]
        unit_id = key["unit_id"]

        outer_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=outer_ensemble_hash) & dict(method_hash=outer_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        inner_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=inner_ensemble_hash) & dict(method_hash=inner_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        
        outer_mei=torch.load(outer_mei_path)
        inner_mei=torch.load(inner_mei_path)

        self.centerimg = outer_mei[0][0]
        self.ring_mask=(outer_mei[0][1] - inner_mei[0][1] > mask_thres_for_ring) * 1

        
    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        randinitial = self._create_random_tensor(*shape)
        initial = randinitial * self.ring_mask + self.centerimg
        return initial

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"

class RandomNormalRing(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""
    """center_type could be 'noise', 'grey','light','null'，'naturalimg' """
    _create_random_tensor = randn

    def __init__(self, key, mask_thres_for_ring=0.3, center_type='null', center_norm=20,img_id=0):
        src_method_fn = key["src_method_fn"]
        unit_id = key["unit_id"]

        inner_ensemble_hash = key["inner_ensemble_hash"]
        outer_ensemble_hash = key["outer_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        outer_method_hash = key["outer_method_hash"]
        unit_id = key["unit_id"]

        outer_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=outer_ensemble_hash) & dict(method_hash=outer_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        inner_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=inner_ensemble_hash) & dict(method_hash=inner_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        
        outer_mei=torch.load(outer_mei_path)
        inner_mei=torch.load(inner_mei_path)

        self.ring_mask = (outer_mei[0][1] - inner_mei[0][1] > mask_thres_for_ring) * 1
        self.inner_mask= (inner_mei[0][1] > 0.3) * 1
        self.center_type = center_type
        self.center_norm = center_norm
        self.img_id = img_id # for the case with natural image in center
    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        randinitial = self._create_random_tensor(*shape)
        if self.center_type == 'noise':
            centerimg = randinitial * self.inner_mask
        if self.center_type == 'grey':
            centerimg = torch.ones(*shape) * -3 * self.inner_mask
        if self.center_type == 'light':
            centerimg = torch.ones(*shape) * 3 * self.inner_mask       
        if self.center_type == 'null':
            centerimg = torch.ones(*shape) * 0
        if self.center_type == 'naturalimg':
            dataset_fn='nndichromacy.datasets.static_loaders'
            dataset_config = {
                'paths': ['/data/mouse/toliaslab/static/static22564-3-12-GrayImageNet-50ba42e98651ac33562ad96040d88513.zip'],
                 'normalize': True,
                 'include_behavior': False,
                 'batch_size': 128,
                 'exclude': None,
                 'file_tree': True,
                  'scale':1}
            dataloaders = builder.get_data(dataset_fn, dataset_config)
            images = []
            for i,j in dataloaders['train']['22564-3-12']:
                images.append(i.squeeze().data)
            centerimg = torch.vstack(images)[self.img_id].cpu() * self.inner_mask   

        if self.center_type != 'null':
            centerimg = centerimg * (self.center_norm / torch.norm(centerimg)) # control centerimg contrast

        initial = centerimg + randinitial * self.ring_mask

        return initial

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"

class RandomNormalSurround(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""
    """center_type could be 'noise', 'grey','light','null' """
    _create_random_tensor = randn

    def __init__(self, key, mask_thres=0.3):
        src_method_fn = key["src_method_fn"]
        unit_id = key["unit_id"]

        inner_ensemble_hash = key["inner_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        unit_id = key["unit_id"]

        inner_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=inner_ensemble_hash) & dict(method_hash=inner_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        
        #outer_mei=torch.load(outer_mei_path)
        inner_mei=torch.load(inner_mei_path)

        self.center_mask= (inner_mei[0][1] > mask_thres) * 1
        #self.center_type = center_type
        #self.center_norm = center_norm
        self.centerimg = inner_mei[0][0]

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        randinitial = self._create_random_tensor(*shape)
        initial = self.centerimg + randinitial * (1-self.center_mask)

        return initial

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"

class RandomNormalCenter(InitialGuessCreator):

    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""
    """center_type could be 'noise', 'grey','light','null' """
    _create_random_tensor = randn

    def __init__(self, key, mask_thres=0.3):
        src_method_fn = key["src_method_fn"]
        unit_id = key["unit_id"]

        inner_ensemble_hash = key["inner_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        unit_id = key["unit_id"]

        inner_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=inner_ensemble_hash) & dict(method_hash=inner_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        
        #outer_mei=torch.load(outer_mei_path)
        inner_mei=torch.load(inner_mei_path)

        self.center_mask= (inner_mei[0][1] > mask_thres) * 1

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        randinitial = self._create_random_tensor(*shape)
        initial = randinitial * (self.center_mask)

        return initial

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class RandomNormBehaviorPositionsSurr(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __init__(self, selected_channels, selected_values, key, mask_thres=0.3):
        if not isinstance(selected_channels, Iterable) and (selected_channels is not None):
            selected_channels = (selected_channels)

        if not isinstance(selected_values, Iterable) and (selected_values is not None):
            selected_values = (selected_values)

        self.selected_channels = selected_channels
        self.selected_values = selected_values
        
        src_method_fn = key["src_method_fn"]
        unit_id = key["unit_id"]

        inner_ensemble_hash = key["inner_ensemble_hash"]
        inner_method_hash = key["inner_method_hash"]
        unit_id = key["unit_id"]

        inner_mei_path = (MEI & dict(method_fn=src_method_fn) & dict(ensemble_hash=inner_ensemble_hash) & dict(method_hash=inner_method_hash) & dict(unit_id=unit_id)).fetch1('mei', download_path=fetch_download_path)
        
        #outer_mei=torch.load(outer_mei_path)
        inner_mei=torch.load(inner_mei_path)

        self.center_mask= (inner_mei[0][-1] > mask_thres) * 1
        #self.center_type = center_type
        #self.center_norm = center_norm
        self.centerimg = inner_mei[0][:2]
        
    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""

        initial = self._create_random_tensor(*shape)
        if self.selected_channels is not None:
            for channel, value in zip(self.selected_channels, self.selected_values):
                initial[:, channel, ...] = value
        initial[:, -2:, ...] = torch.from_numpy(np.stack(np.meshgrid(np.linspace(-1, 1, shape[-1]), np.linspace(-1, 1, shape[-2]))))
        initial[:,:2, ...] = self.centerimg + initial[:,:2,...] * (1-self.center_mask)
        return initial

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"

