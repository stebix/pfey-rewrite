from functools import partial
import numpy as np
import torch

from functools import partial
from typing import Sequence, Iterable
from torchvision import transforms

from dataload import FileRecord, load_filerecord_training
from utils import expand_4D


def compute(candidate, array):
    """
    Compute candidate with array argument if candidate is callable. Otherwise
    try to recurse into dictionaries."""
    if callable(candidate):
        return candidate(array)
    elif isinstance(candidate, dict):
        return {key : compute(value, array) for key, value in candidate.items()}
    else:
        raise TypeError(f'cannot execute for {type(candidate)}')


def compute_descriptive_statistics(dataset: torch.utils.data.Dataset,
                                   q_values: Iterable[float] = (0.75, 0.95, 0.99, 0.999)) -> dict:
    """
    Compute a selection of descriptive statistical parameters on the provided
    dataset.

    Since the full dataset is accumulated in-memory, this function can have a
    quite large peak memory footprint.

    Parameters
    ----------

    dataset : torch.utils.data.Dataset
    
    q_values : Iterable of float
        The quantile values for which the corresponding parameter
        is computed. Must be a sequence of float in [0, 1].
    
    Returns
    -------

    parameter_values : Dict[str, Union[float, dict]]
        The statistical parameter computation result.
        Quantile computation results are contained in a nested subdict.
    """
    parameter_functions = {
        'min' : np.min, 'max' : np.max, 'mean' : np.mean, 'median' : np.median,
        'quantiles' : {str(q_value) : partial(np.quantile, q=q_value)
                       for q_value in q_values}
    }
    spectra = []
    for item in dataset:
        data, _ = item
        spectra.append(data)
    spectra = np.stack(spectra)
    parameter_values = {key : compute(value, spectra)
                        for key, value in parameter_functions.items()}
    return parameter_values




class CellDataset(torch.utils.data.TensorDataset):
    """
    Dataset encapsulating data for a single cell type.
    """
    labelstyle: str = 'classindex'

    @classmethod
    def from_filerecord(cls, filerecord: FileRecord, max_augcount: int) -> 'CellDataset':
        """Instantiate directly from celltype specific, directory-describing FileRecord"""
        data, label = load_filerecord_training(filerecord, max_augcount, cls.labelstyle)
        data = expand_4D(data, dim='C')
        # The np.int32 recast is crucial: torch principally does not allow uint16
        data = torch.as_tensor(data.astype(np.int32))
        label = torch.as_tensor(label.astype(np.int32))
        return cls(data, label)



class TrainingDataset(torch.utils.data.ConcatDataset):
    """
    Assemble full training dataset comprising several `CellDatasets` for
    different cell types.
    """
    @classmethod
    def from_filerecords(cls, filerecords: Sequence[FileRecord],
                         max_augcount: int) -> 'TrainingDataset':
        """
        Create `CellDataset` from sequence of FileRecords and concat towards
        single TrainingDataset.

        Parameters
        ----------

        filerecords : Sequence of FileRecord
            Contains compound information about celltype and data load paths.

        max_augcount : int
            Global limit of augmentations to include into dataset. At most
            this many augmented spectra are loaded.

        Returns
        -------

        TrainingDataset
            The fully concatenated dataset, in memory.
        """
        datasets = [
            CellDataset.from_filerecord(filerecord, max_augcount)
            for filerecord in filerecords
        ]
        return cls(datasets)


class TransformedTrainingDataset(TrainingDataset):
    """
    Automatically preprocess data via composed transformer.

    Parameters
    ----------

    datasets : Iterable of torch.data.utils.Dataset
        The dataset iterable used to create the full dataset
        via concatenation.

    transformer : transforms.Compose
        The `Compose` instance holding the sequence of data
        transformations.
    """
    def __init__(self,
                 datasets: Iterable[torch.utils.data.Dataset],
                 transformer: transforms.Compose) -> None:
        super().__init__(datasets)
        self.transformer = transformer

    def __getitem__(self, idx: int) -> tuple:
        (data, label) = super().__getitem__(idx)
        data = self.transformer(data)
        return (data, label)

    @classmethod
    def from_filerecords(cls,
                         filerecords: Sequence[FileRecord],
                         max_augcount: int,
                         transformer: transforms.Compose) -> 'TransformedTrainingDataset':
        """
        Create `CellDataset` instances from sequence of FileRecords and concat towards
        single TransformedTrainingDataset. Upon item request, the transformations contained
        in the `Compose` instance are applied.

        Parameters
        ----------

        filerecords : Sequence of FileRecord
            Contains compound information about celltype and data load paths.

        max_augcount : int
            Global limit of augmentations to include into dataset. At most
            this many augmented spectra are loaded.

        transformer : transforms.Compose
            Sequential composition of various data transformations.

        Returns
        -------

        TrainingDataset
            The fully concatenated dataset, in memory.
        """
        datasets = [
            CellDataset.from_filerecord(filerecord, max_augcount)
            for filerecord in filerecords
        ]
        return cls(datasets, transformer)
    