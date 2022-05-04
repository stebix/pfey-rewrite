import warnings
import numpy as np
import random

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List
from collections import defaultdict, namedtuple

import matplotlib.pyplot as plt
import imageio


from datatools import create_onehot_label


TrainingSample = namedtuple(typename='TrainingSample',
                            field_names=['data', 'label'])


@dataclass
class FileRecord:
    celltype: str
    int_ID: int
    path: Path
    augmentations: Dict[int, Path]



def display():
    """Preliminary cell display"""
    p = Path('C:/Users/Jannik/Desktop/pfay-rewrite/data/Aug30_5Stretch_5Shift_woMSCs/5/L929-6_7.png')
    assert p.is_file()

    img = imageio.imread(p)

    fig, ax = plt.subplots()
    ax.hist(img.flatten(), bins=50)
    ax.set_yscale('log')
    plt.show()



def load_png_as_numpy(filepath: Path) -> np.ndarray:
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
    Load the PNG files from a directory containing data for a single
    cell type.
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
    
    # return filemap
    
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

        





def load_filerecord_training(filerecord: FileRecord,
                             max_augcount: int) -> List[Tuple[np.ndarray]]:
    """
    Load original and augmented spectrum data from filerecord for training purposes
    and pair it with corresponding onehot label vectors.

    Parameters
    ----------

    filerecord : FileRecord
        The compound data storing the files corresponding to a single UIID.

    max_augcount : int
        Limit the number of loaded augmentations to at most this integer.
    """
    labelvector = create_onehot_label(filerecord.celltype)
    spectrum_data = [load_png_as_numpy(filerecord.path)]
    # Compute actual count of loaded augmentations.
    augcount = min(len(filerecord.augmentations), max_augcount)
    spectrum_data.extend((
        load_png_as_numpy(augpath)
        for augpath in random.sample(list(filerecord.augmentations.values()), k=augcount)
    ))
    spectrum_data = np.stack(spectrum_data, axis=0)
    broadcast_shape = (spectrum_data.shape[0], labelvector.shape[0])
    labelvector = np.broadcast_to(labelvector[np.newaxis, :],
                                  shape=broadcast_shape)
    return TrainingSample(data=spectrum_data, label=labelvector)




        
        


def load_celldata(directory, logger):
    """
    Load cell scan data. Assumes that data is ordered into subdirectories corresponding to
    a certain cell type. Based on file name, the `FileRecord` object for a coherent collection
    of files is generated.
    """
    directory = Path(directory)
    for candidate in directory.iterdir():
        if not candidate.is_dir():
            print(f'Skipping element: {candidate} :: NotADirectory')
            continue

        filemap = load_subdir(candidate)
        



    


def main():
    p = Path('C:/Users/Jannik/Desktop/pfay-rewrite/data/Aug30_5Stretch_5Shift_woMSCs/5/')
    mapping = load_subdir(p)
    print(mapping[-1])




if __name__ == '__main__':
    main()