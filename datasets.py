import numpy as np
import torch

from typing import Sequence

from dataload import FileRecord, load_filerecord_training



class CellDataset(torch.utils.data.TensorDataset):
    """
    Dataset encapsulating data for a single cell type.
    """
    @classmethod
    def from_filerecord(cls, filerecord: FileRecord, max_augcount: int) -> 'CellDataset':
        """Instantiate directly from celltype specific, directory-describing FileRecord"""
        data, label = load_filerecord_training(filerecord, max_augcount)
        # Recast as torch.Tensor from np.int32 recast: torch does not allow uint16
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