import numpy as np

from typing import Optional

class Cropper:
    """
    Crop N-d arrays of the layout (... x H x W) by cutting elements with indices
    outside of the indicated height or width range.
    Second-to-last axis/dimension is interpreted as height dimension.
    Last axis/dimension is interpreted as width dimension.
    Visual explanation:

        w_start_idx  w_stop_idx

               ┌──┬─────┬───┐
               │  │     │   │
               │  │     │   │
    h_start_idx├──┼─────┼───┤  ┌─────┐
               │  │xxxxx│   │  │xxxxx│
               │  │xxxxx│   │  │xxxxx│  <- Crop result
               │  │xxxxx│   │  │xxxxx│     with surviving region
               │  │xxxxx│   │  │xxxxx│     
               │  │xxxxx│   │  │xxxxx│
    h_stop_idx ├──┼─────┼───┤  └─────┘
               │  │     │   │
               │  │     │   │
               └──┴─────┴───┘
    
    Parameters
    ----------

    h_start_index : int, optional
        Set the lower row index of the surviving region.
        Rows with lower indices are cropped.
        Defaults to 0.

    h_stop_index: int or None, optional
        Set the upper row index of the surviving region.
        Rows with bigger indices are cropped.
        Defaults to None (i.e. last row). 
    
    w_start_index : int, optional
        Set the lower column index of the surviving region.
        Columns with lower indices are cropped.
        Defaults to 0.

    w_stop_index: int or None, optional
        Set the upper column index of the surviving region.
        Columns with bigger indices are cropped.
        Defaults to None (i.e. last column). 

    """
    def __init__(self,
                 h_start_index: int = 0, h_stop_index: Optional[int] = None,
                 w_start_index: int = 0, w_stop_index: Optional[int] = None) -> None:
        self.h_start_index = h_start_index
        self.h_stop_index = h_stop_index
        self.w_start_index = w_start_index
        self.w_stop_index = w_stop_index


    def __str__(self) -> str:
        attribute_str = (f'h_start_index={self.h_start_index}, h_stop_index={self.h_stop_index}, '
                         f'w_start_index={self.w_start_index}, w_stop_index={self.w_stop_index}')
        return f'{self.__class__.__name__}({attribute_str})'
    
    def __call__(self, array: np.ndarray) -> np.ndarray:
        inert_axes_count = len(array.shape) - 2
        inert_axes = tuple(np.s_[:] for _ in range(inert_axes_count))
        indices = ((self.h_start_index, self.h_stop_index),
                   (self.w_start_index, self.w_stop_index))
        crop_axes = tuple(np.s_[start:stop] for start, stop in indices)
        indices = (*inert_axes, *crop_axes)
        return array[indices]



class Standardizer:
    """
    Simple standardization via division by the predefined `cval`.
    """
    def __init__(self, cval: float) -> None:
        self.cval = cval
    
    def __call__(self, array: np.ndarray) -> np.ndarray:
        return array / self.cval

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(cval={self.cval})'

    def __repr__(self) -> str:
        return self.__str__()



if __name__ == '__main__':
    import scipy.misc
    import matplotlib.pyplot as plt

    shape = (2, 1, 333, 333)
    array = np.random.default_rng().integers(size=shape, low=0, high=10)
    image = scipy.misc.face()

    cropper = Cropper(137, 137)
    print(cropper)

    arr_result = cropper(array)
    # this image has H x W x C layout
    img_result = cropper(np.swapaxes(image, 0, -1))

    # restore H x W x C layout
    img_result = np.swapaxes(img_result, 0, -1)

    print(f'Array crop result shape: {arr_result.shape}')
    print(f'Image crop result shape: {img_result.shape}')
    print('Image result visualization:')
    fig, axes = plt.subplots(ncols=2)
    axes = axes.flat
    ax = axes[0]
    ax.set_title('Original')
    ax.imshow(image)

    ax = axes[1]
    ax.set_title('Cropped')
    ax.imshow(img_result)

    plt.show()

