import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from sklearn.metrics import confusion_matrix

from typing import Union, Optional, Dict
from collections import OrderedDict


def plot_tally(tallydict: dict):
    # impose static ordering
    tallydict = OrderedDict(tallydict)
    # compute plotting data
    x = np.arange(len(tallydict))
    int_ID_counts = [subdict['count'] for subdict in tallydict.values()]
    augmentation_counts = [subdict['augmentations'] for subdict in tallydict.values()]
    xticklabels = list(tallydict.keys())
    fig, axes = plt.subplots(ncols=2, figsize=(9,4))
    axes = axes.flat
    # plot for the unique integer IDs
    ax = axes[0]
    ax.bar(x, int_ID_counts, tick_label=xticklabels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.set_ylabel('Items $N$')
    ax.set_title('Unique integer ID count')
    # plot for the augmentation count
    ax = axes[1]
    ax.bar(x, augmentation_counts, tick_label=xticklabels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.set_title('Augmentation count')
    plt.tight_layout()


def plot_confusion_matrix(predictions: Union[list, np.ndarray],
                          labels: Union[list, np.ndarray],
                          classindex_to_label_mapping: Dict[int, str],
                          normalize: Optional[str] = 'all') -> matplotlib.figure.Figure:
    """
    Parameters
    ----------
    
    predictions : list or np.ndarray
        The collection of predictions. If given a list,
        a concatenation is performed to create a single
        prediction vector.
        
    labels : list or np.ndarray
        The collection of corresponding labels. If given a list,
        a concatenation is performed to create a single
        label vector.
        
    classindex_to_label_mapping : dict
        Mapping from classindex integer to label string.
        Is utilized to produce axis tick labels.
    
    normalize : str or None, optional
        Normalizes confusion matrix over the true (rows), predicted
        (columns) conditions or all the population.
        Can be {'true', 'pred', 'all'}.
        If None, confusion matrix will not be normalized.
        Defaults to None.
    """
    if isinstance(predictions, list):
        predictions = np.concatenate(predictions)
    if isinstance(labels, list):
        labels = np.concatenate(labels)
    conf_mat = confusion_matrix(labels, predictions, normalize=normalize)
    celltype_indices = np.unique(labels)
    celltype_names = [classindex_to_label_mapping[idx] for idx in celltype_indices]
    conf_dfr = pd.DataFrame(conf_mat, columns=celltype_names, index=celltype_names)
    # produce plot conveniently from dataframe
    fig, ax = plt.subplots()
    sn.heatmap(conf_dfr, ax=ax, annot=True, xticklabels=True)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    return fig