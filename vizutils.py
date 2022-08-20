import numpy as np
import matplotlib.pyplot as plt

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