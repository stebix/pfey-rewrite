from itertools import chain
import warnings
import numpy as np
import random

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, Dict, List
from collections import defaultdict, namedtuple

import imageio

from datatools import create_onehot_label, create_classindex_label
from utils import train_test_split, Partition


TrainingSample = namedtuple(typename='TrainingSample',
                            field_names=['data', 'label'])


@dataclass
class FileRecord:
    """
    Compound data record that encapsulates the information and paths
    corresponding to a single integer ID.
    
    Attributes
    ----------

        celltype : str
            identifier for cell type

        int_ID : int
            Globally unique integer ID
        
        path : Path
            The filepath pointing to original spectrum data

        augmentations : dict
            Mapping from augmentation integer ID to path
            to augmented spectrum data.  
    """
    celltype: str
    int_ID: int
    path: Path
    augmentations: Dict[int, Path]


def celltype_aware_train_test_split(filerecords: Sequence[FileRecord],
                                    train_fraction: float) -> Tuple[List[FileRecord]]:
    """
    Train-test split a sequence of filerecords such that each each cell type is split with
    the indicated training fraction such that (modulo rounding):
    ->    `training_set = items * training_fraction` and `test_set = items * (1 - training_fraction)`
    holds.

    Parameters
    ----------

    filerecords : sequence of FileRecords
        A sequence containing all file record instances.

    train_fraction : float
        The fraction of the items selected per cell type going
        into the training set.
        Must be a float value between 0 and 1.
    """
    celltype_mapping = defaultdict(list)
    train = []
    test = []
    for filerecord in filerecords:
        celltype_mapping[filerecord.celltype].append(filerecord)
    for filerecords in celltype_mapping.values():
        partition = train_test_split(filerecords, train_fraction)
        train.extend(partition.train)
        test.extend(partition.test)
    return Partition(train=train, test=test)


def tally_filerecords(filerecords: Sequence[FileRecord]) -> dict:
    """
    Sort and count the filerecords depending on their assigned cell type field.
    """
    infodict = {}
    for filerecord in filerecords:
        try:
            subinfodict = infodict[filerecord.celltype]
        except KeyError:
            infodict[filerecord.celltype] = {
                'int_IDs' : [filerecord.int_ID],
                'augmentations' : len(filerecord.augmentations)
            }
            continue
        subinfodict['int_IDs'].append(filerecord.int_ID)
        subinfodict['augmentations'] += len(filerecord.augmentations)
    # check for duplicate integer IDs
    joined_int_IDs = []
    for celltype, subinfodict in infodict.items():
        ID_list = subinfodict['int_IDs']
        ID_set = set(ID_list)
        if len(ID_list) != len(ID_set):
            warnings.warn(f'duplicate integer ID for cell type {celltype} detected')
        joined_int_IDs.extend(ID_list)
        subinfodict['count'] = len(subinfodict['int_IDs'])
    joined_int_ID_set = set(joined_int_IDs)
    if len(joined_int_IDs) != len(joined_int_ID_set):
        warnings.warn(f'duplicate integer ID between different cell types detected')
    return infodict



    

def load_png_as_numpy(filepath: Path) -> np.ndarray:
    """
    Load a PNG file from the indicated filepath as a pure numpy.ndarray.
    """
    if not filepath.name.endswith('png'):
        warnings.warn(
            f'attempting to load PNG from path "{filepath.resolve()}": '
            f'incongruent file extension {filepath.suffix}'
        )
    image = imageio.imread(filepath)
    return np.asarray(image)



def parse_filename(filename: str, check_suffix: bool = False) -> dict:
    """
    Parse file information from the name and return information
    as a dictionary.
    Custom function for Fey project naming scheme.
    """
    valid_suffixes = set(['png', 'PNG'])
    stem, suffix = filename.split('.')

    if check_suffix:
        if not suffix in valid_suffixes:
            raise ValueError(
                f'Invalid suffix: {suffix}, Must be in {valid_suffixes}!'
            )

    celltype, series_info = stem.split('-')
    int_ID, augmentation_state = series_info.split('_')
    info = {'celltype' : celltype, 'int_ID' : int(int_ID),
            'augmentation_state' : augmentation_state}
    return info


def load_subdir(directory: Path) -> List[FileRecord]:
    """
    Load the PNG files from a directory as a list of filerecords that
    provide a cohesive object containing information for a single
    integer ID and celltype.
    Hints:
        - Subdirectories or 'Non-files' are skipped
        - Files without 'png' suffix are skipped
    
    Parameters
    ----------

    directory: Path


    Returns
    -------

    filerecords : List of FileRecords
        The parsed contents of the directory
        as list of FileRecord instances


    Raises
    ------

    RuntimeError
        The function expects that only files for a single celltype are
        present in the loaded subdirectory. RuntimeError is raised if
        multiple celltypes are encountered.
    
    """
    encountered_celltypes = set()
    filemap = defaultdict(dict)
    for item in directory.iterdir():
        # handle special cases for skipping
        if not item.is_file():
            print(f'Skipping {item} :: Not a file')
            continue
        if not item.name.endswith('png'):
            print(f'Skipping {item} :: No PNG file suffix')
            continue
        info = parse_filename(item.name)
        # keep track of celltypes encountered in this directory
        encountered_celltypes.add(info['celltype'])
        if len(encountered_celltypes) > 1:
            raise RuntimeError(
                f'Tainted directory: Found multiple celltypes: {encountered_celltypes}'
            )
        filemap[info['int_ID']][info['augmentation_state']] = item 
    # retrieve the celltype (enforced single specific celltype in this subdirectory)
    celltype = encountered_celltypes.pop()
    filerecords = []
    for int_ID, variations in filemap.items():
        opath = variations.pop('Original')
        filerecords.append(
            FileRecord(celltype=celltype, int_ID=int_ID,
                       path=opath, augmentations=variations)
        )
    return filerecords


def load_directory(directory: Path) -> List[FileRecord]:
    all_filerecords = []
    for element in directory.iterdir():
        if element.is_dir():
            all_filerecords.extend(load_subdir(element))
        else:
            continue
    return all_filerecords


def load_filerecord_training(filerecord: FileRecord,
                             max_augcount: int,
                             labelstyle: str) -> List[Tuple[np.ndarray]]:
    """
    Load original and augmented spectrum data from filerecord for training purposes
    and pair it with corresponding onehot label vectors.

    Parameters
    ----------

    filerecord : FileRecord
        The compound data storing the files corresponding to a single UIID.

    max_augcount : int
        Limit the number of loaded augmentations to at most this integer.

    labelstyle : str
        Determines the style of the produced label. May be 'onehot' for
        a onehot vector or 'classindex' for a class index singleton array.
    """
    if labelstyle == 'onehot':
        create_label = create_onehot_label
    elif labelstyle == 'classindex':
        create_label = create_classindex_label
    else:
        message = (f'Invalid labelstyle "{labelstyle}" argument. Must be '
                   f'"onehot" or "classindex"')
        raise ValueError(message)
    label = create_label(filerecord.celltype)
    spectrum_data = [load_png_as_numpy(filerecord.path)]
    # Compute actual count of loaded augmentations.
    augcount = min(len(filerecord.augmentations), max_augcount)
    spectrum_data.extend((
        load_png_as_numpy(augpath)
        for augpath in random.sample(list(filerecord.augmentations.values()), k=augcount)
    ))
    spectrum_data = np.stack(spectrum_data, axis=0)
    broadcast_shape = (spectrum_data.shape[0], label.shape[0])
    label = np.broadcast_to(label[np.newaxis, :],
                            shape=broadcast_shape)
    return TrainingSample(data=spectrum_data, label=label)


def load_filerecord_test(filerecord: FileRecord) -> Tuple[np.ndarray]:
    """
    Load data from a FileRecord instance for testing/validation purposes.
    This means that only the original, non-augmented data is read.
    """
    labelvector = create_onehot_label(filerecord.celltype)
    spectrum_data = load_png_as_numpy(filerecord.path)
    return TrainingSample(data=spectrum_data, label=labelvector)



def main():
    for i in range(1, 10):
        p = Path(f'C:/Users/Jannik/Desktop/pfay-rewrite/data/Aug30_5Stretch_5Shift_woMSCs/{i}/')
        mapping = load_subdir(p)
        print(f'Counter :: {i}')
        print(f'Found {len(mapping)} unique cell datasets')




if __name__ == '__main__':
    main()