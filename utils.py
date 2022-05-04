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


def expand_4D(array_like: Union[np.ndarray, torch.Tensor]):
    if isinstance(array_like, np.ndarray):
        return expand_4D_numpy(array_like)
    elif isinstance(array_like, torch.Tensor):
        return expand_4D_torch(array_like)
    else:
        message = f'cannot expand object of type {type(array_like)} to 4D'
        raise TypeError(message)


def expand_4D_torch(tensor: torch.Tensor) -> torch.Tensor:
    """
    Interpret 2D or 3D tensors as (H x W) and (N x H x W) respectively
    and expand to 4D (N x C x H x W) via adding of fake channel dimension.
    """
    if tensor.ndim == 2:
        tensor = torch.unsqueeze(torch.unsqueeze(tensor))
    elif tensor.ndim == 3:
        tensor = torch.unsqueeze(tensor)
    else:
        message = f'tensor with ndim = {tensor.ndim} note eligible for 4D expansion'
        raise ValueError(message)
    return tensor


def expand_4D_numpy(array: np.ndarray) -> np.ndarray:
    """
    Interpret 2D or 3D arrays as (H x W) and (N x H x W) respectively
    and expand to 4D (N x C x H x W) via adding of fake channel dimension.
    """
    if array.ndim == 2:
        array = array[np.newaxis, np.newaxis, ...]
    elif array.ndim == 3:
        array = array[np.newaxis, ...]
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




if __name__ == '__main__':
    _ = get_nonlinearity('ReLU')