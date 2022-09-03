import numpy as np

from collections import OrderedDict

# project specific global cell types
CELL_TYPES = ('CHO', 'K562', 'C2C12', 'THP1',
              'HeLa', 'MDA231', 'Vero', 'L929',
              'A549','HEK293T')

CELL_TYPES_MAPPING = OrderedDict([
    ('CHO', 0),
    ('K562', 1),
    ('C2C12', 2),
    ('THP1', 3),
    ('HeLa', 4),
    ('MDA231', 5), 
    ('Vero', 6),
    ('L929', 7),
    ('A549', 8),
    ('HEK293T', 9)
])

CLASSIDX_TO_CELLTYPE = OrderedDict(
    (index, name) for name, index in CELL_TYPES_MAPPING.items()
)

def onehot(length: int, hotpos: int) -> np.ndarray:
    """Create a onehot vector."""
    vector = np.zeros(length)
    vector[hotpos] = 1
    return vector


def create_onehot_label(celltype: str) -> np.ndarray:
    """
    Create a one-hot label vector from the cell type string.
    Relies on module global constant `CELL_TYPES` for ordering.
    """
    if celltype not in CELL_TYPES_MAPPING.keys():
        raise ValueError(f'invalid cell type string "{celltype}"')
    return onehot(len(CELL_TYPES_MAPPING), CELL_TYPES_MAPPING[celltype])


def create_classindex_label(celltype: str) -> np.ndarray:
    """
    Create a class-index based label: a singleton array containing
    the class index 
    """
    if celltype not in CELL_TYPES_MAPPING.keys():
        raise ValueError(f'invalid cell type string "{celltype}"')
    return np.array([CELL_TYPES_MAPPING[celltype]])


def celltype_from_onehot(onehot_vector: np.ndarray) -> str:
    """
    Map to cell type string from one-hot label vector.
    Relies on module global constant `CELL_TYPES` for ordering.
    """
    return CELL_TYPES[np.argmax(onehot_vector)]


def celltype_from_classindex(classindices: np.ndarray):
    inverted_mapping = {
        index : celltype
        for celltype, index in CELL_TYPES_MAPPING.items()
    }
    celltypes = []
    for classindex in classindices:
        celltypes.append(inverted_mapping[classindex])
    return celltypes


