import importlib
import numpy as np
import random

from math import floor
from collections import namedtuple
from typing import Sequence, Union

import torch


Partition = namedtuple(typename='Partition', field_names=['train', 'test'])


def train_test_split(items: Sequence, train_fraction: float) -> Partition:
    """
    Split a sequence of items into two tuples based on a train fraction.
    The test fraction is 1 - train fraction.
    The first element of the `Partition` tuple denotes the training set, the
    second element the test set.
    """
    if train_fraction > 1.0 or train_fraction < 0.0:
        message = f'invalid train fraction {train_fraction} : must be betweeen 0 and 1'
        raise ValueError(message)
    # Perform in-place shuffling. 
    random.shuffle(list(items))
    # Select the split from the items list.
    n_train = floor(len(items) * train_fraction)
    return Partition(train=items[:n_train], test=items[n_train:])


def expand_4D(array_like: Union[np.ndarray, torch.Tensor], dim: str = 'N'):
    """
    Expand a 2D or 3D numpy array or torch tensor to the canonical
    4D layout (N x C x H x W).

    Parameters
    ----------

    array_like : np.ndarray or torch.Tensor
        The array-like object that is expanded to 4D
    
    dim : str, optional
        Selects the single expanded dimension, either
        batch ('N') or channel ('C') in the case of a
        3D `array_like` input. Has no effect for 2D inputs.
        Defaults to 'N'. 
    """
    if isinstance(array_like, np.ndarray):
        return expand_4D_numpy(array_like, dim)
    elif isinstance(array_like, torch.Tensor):
        return expand_4D_torch(array_like, dim)
    else:
        message = f'cannot expand object of type {type(array_like)} to 4D'
        raise TypeError(message)


def expand_4D_torch(tensor: torch.Tensor, dim: str) -> torch.Tensor:
    """
    Interpret 2D or 3D tensors as (H x W) and (N x H x W) respectively
    and expand to 4D (N x C x H x W) via adding of fake channel dimension.

    For 3D tensor inputs, `fake_dim` selects to position of the expansion
    Use 'C' for channel expansion at dim 1: (N x H x W)  ->  (N x C x H x W)
    Use 'N' for batch expansion at dim 0:   (C x H x W)  ->  (N x C x H x W)
    """
    if tensor.ndim == 2:
        tensor = torch.unsqueeze(torch.unsqueeze(tensor))
    elif tensor.ndim == 3:
        if dim == 'C':
            dim = 1
        elif dim == 'N':
            dim = 0
        else:
            raise ValueError(f'invalid dim "{dim}", expected "N" or "C"')
        tensor = torch.unsqueeze(tensor, dim=dim)
    else:
        message = f'tensor with ndim = {tensor.ndim} note eligible for 4D expansion'
        raise ValueError(message)
    return tensor


def expand_4D_numpy(array: np.ndarray, dim: str) -> np.ndarray:
    """
    Interpret 2D or 3D arrays as (H x W) and (N x H x W) respectively
    and expand to 4D (N x C x H x W) via adding of fake channel dimension.

    For 3D array inputs, `fake_dim` selects to position of the expansion
    Use 'C' for channel expansion at dim 1: (N x H x W)  ->  (N x C x H x W)
    Use 'N' for batch expansion at dim 0:   (C x H x W)  ->  (N x C x H x W)
    """
    if array.ndim == 2:
        array = array[np.newaxis, np.newaxis, ...]
    elif array.ndim == 3:
        if dim == 'N':
            slc = np.s_[np.newaxis, ...]
        elif dim == 'C':
            slc = np.s_[:, np.newaxis, :, :]
        else:
            raise ValueError(f'invalid dim "{dim}", expected "N" or "C"')
        array = array[slc]
    else:
        message = f'array with ndim = {array.ndim} note eligible for 4D expansion'
        raise ValueError(message)
    return array


def get_nonlinearity(nonlinearity: str, **kwargs) -> torch.nn.Module:
    """
    Get a nonlinearity callable from its canonical name string.
    """
    module = importlib.import_module(name='torch.nn')
    nonlinearty_class = getattr(module, nonlinearity)
    return nonlinearty_class(**kwargs)


def recursive_to_tensor(candidate):
    """
    Attempt to recast a candidate object to `torch.Tensor`
    If candidate is a (nested) tuple or list, elements are recursively
    cast to tensor.
    """
    if isinstance(candidate, (list, tuple)):
        return [recursive_to_tensor(elem) for elem in candidate]
    elif isinstance(candidate, np.ndarray):
        return torch.from_numpy(candidate)
    elif isinstance(candidate, torch.Tensor):
        return candidate
    else:
        message = f'could not cast type {type(candidate)} to tensor'
        raise RuntimeError(message)

