import numpy as np


# project specific global cell types
CELL_TYPES = ('CHO', 'K562', 'C2C12', 'THP1',
              'HeLa', 'MDA231', 'Vero', 'L929',
              'A549','HEK293T')


CELL_TYPES_ALT = ('HeLa', 'Vero', 'THP1')



def create_onehot_label(celltype: str) -> np.ndarray:
    """
    Create a one-hot label vector from the cell type string.
    Relies on module global constant `CELL_TYPES` for ordering.
    """
    if celltype not in CELL_TYPES:
        raise ValueError(f'invalid cell type string "{celltype}"')
    cells = np.array(CELL_TYPES)
    return np.where(cells == celltype, 1, 0)


def celltype_from_onehot(onehot_vector: np.ndarray) -> str:
    """
    Map to cell type string from one-hot label vector.
    Relies on module global constant `CELL_TYPES` for ordering.
    """
    return CELL_TYPES[np.argmax(onehot_vector)]


