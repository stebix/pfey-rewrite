import warnings
import torch
import numpy as np

from collections import OrderedDict
from typing import Sequence, Union, Tuple, Optional

from utils import get_nonlinearity

NORM_MAPPING = {
    'bnorm' : torch.nn.BatchNorm2d,
    'inorm' : torch.nn.InstanceNorm2d,
    'gnorm' : torch.nn.GroupNorm
}

ACTIVATION_MAPPING = {
    'relu' : torch.nn.ReLU,
    'lrelu' : torch.nn.LeakyReLU,
    'prelu' : torch.nn.PReLU,
    'gelu' : torch.nn.GELU,
    'elu' : torch.nn.ELU,
    'celu' : torch.nn.CELU,
    'softmax' : torch.nn.Softmax,
    'sigmoid' : torch.nn.Sigmoid,
    'identity' : torch.nn.Identity
}

POOLING_MAPPING = {
    'avgpool' : torch.nn.AvgPool2d,
    'maxpool' : torch.nn.MaxPool2d
}

ADAPTIVE_POOLING_MAPPING = {
    'adaptiveavgpool' : torch.nn.AdaptiveAvgPool2d,
    'adaptivemaxpool' : torch.nn.AdaptiveMaxPool2d
}


def create_norm(string_alias: str, num_features: int, **kwargs) -> Tuple[str, torch.nn.Module]:
    """
    Create a normalization layer from a string alias and
    arbitrary keywords.

    Parameters
    ----------

    string_alias : str
        Norm type string alias.

    num_features : int
        Norm layer input features.

    kwargs
        Additional keyword arguments.

    """
    string_alias = string_alias.lower()
    strict_kwargs = {'num_features' : num_features}
    if string_alias == 'gnorm':
        strict_kwargs = {'num_channels' : num_features}
        if 'num_groups' not in kwargs:
            raise TypeError(
                f'GroupNorm module requires specification num_groups in '
                f'keyword arguments! Got < {kwargs} > as kwargs!'
            )
    try:
        norm_class = NORM_MAPPING[string_alias]
    except KeyError:
        raise ValueError(
            f'Unrecognized norm string alias "{string_alias}". '
            f'Must be one of {set(NORM_MAPPING.keys())}')
    return ('norm', norm_class(**strict_kwargs, **kwargs))


def create_activation(string_alias: str, **kwargs) -> Tuple[str, torch.nn.Module]:
    """
    Create a nonlinearity or activation function from a string alias and
    arbitrary keywords.
    """
    string_alias = string_alias.lower()
    try:
        activation_class = ACTIVATION_MAPPING[string_alias]
    except KeyError:
        raise RuntimeError(
            f'Unrecognized nonlinearity string alias "{string_alias}"! Must be '
            f'one of {set(ACTIVATION_MAPPING.keys())}'
        )
    return ('activation', activation_class(**kwargs))


def create_pooling(string_alias: str, kernel_size: Union[int, Tuple[int]],
                   **kwargs) -> torch.nn.Module:
    string_alias = string_alias.lower()
    try:
        pooling_class = POOLING_MAPPING[string_alias]
    except KeyError:
        raise RuntimeError(
            f'Unrecognized pooling operation string alias "{string_alias}"! Must be '
            f'one of {set(POOLING_MAPPING.keys())}'
        )
    return pooling_class(kernel_size=kernel_size, **kwargs)


def create_adaptive_pooling(string_alias: str, output_size: Union[int, Tuple[int]],
                            **kwargs) -> torch.nn.Module:
    string_alias = string_alias.lower()
    try:
        pooling_class = ADAPTIVE_POOLING_MAPPING[string_alias]
    except KeyError:
        raise RuntimeError(
            f'Unrecognized adaptive pooling operation string alias "{string_alias}"! Must be '
            f'one of {set(ADAPTIVE_POOLING_MAPPING.keys())}'
        )
    return pooling_class(output_size=output_size, **kwargs)



def create_downsampler(mode: str,
                       kernel_size: Union[int, Tuple[int]],
                       in_channels: Optional[Union[int, Tuple[int]]],
                       out_channels: Optional[Union[int, Tuple[int]]],
                       stride: Union[int, Tuple[int]] = 1,
                       padding: Union[int, Tuple[int]] = 0) -> Tuple[str, torch.nn.Module]:
    pass


def determine_conv_is_primary(blockspec: str) -> bool:
    """
    Determine if convolution operations primary by analyzing the internal
    ordering of convolution, normalization and activation inside the blockspec.
    """
    parts = blockspec.split(',')
    normindex = None
    for normalias in NORM_MAPPING.keys():
        try:
            normindex = parts.index(normalias)
        except ValueError:
            pass
    try:
        convindex = parts.index('conv')
    except ValueError as e:
        msg = f'No convolution position defined in blockspec "{blockspec}"'
        raise ValueError(msg) from e
    conv_first = convindex < normindex
    return conv_first


def create_conv_block_elements(in_channels: int, out_channels: int,
                               kernel_size: Union[int, Tuple[int]],
                               blockspec: str,
                               padding: Union[int, Tuple[int]],
                               padding_mode: str,
                               activation_kwargs: Optional[dict] = None,
                               norm_kwargs: Optional[dict] = None,
                               conv_kwargs: Optional[dict] = None) -> list:
    activation_kwargs = activation_kwargs or {}
    norm_kwargs = norm_kwargs or {}
    conv_kwargs = conv_kwargs or {}
    modules = []
    conv_is_primary = determine_conv_is_primary(blockspec)
    for item in blockspec.split(','):
        if item == 'conv':
            conv = torch.nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, padding=padding,
                padding_mode=padding_mode, **conv_kwargs
            )
            modules.append(('convolution', conv))
        elif item in NORM_MAPPING.keys():
            num_features = out_channels if conv_is_primary else in_channels
            modules.append(create_norm(item, num_features=num_features,
                                       **norm_kwargs))
        elif item in ACTIVATION_MAPPING.keys():
            modules.append(create_activation(item, **activation_kwargs))
        else:
            raise ValueError(f'Unrecognized blockspec item "{item}"')
    return modules



def create_conv_block(in_channels: int, out_channels: int,
                      kernel_size: Union[int, Tuple[int]],
                      blockspec: str,
                      padding: Union[int, Tuple[int]],
                      padding_mode: str,
                      activation_kwargs: Optional[dict] = None,
                      norm_kwargs: Optional[dict] = None,
                      conv_kwargs: Optional[dict] = None) -> list:
    elements = create_conv_block_elements(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
        blockspec=blockspec, padding=padding, padding_mode=padding_mode,
        activation_kwargs=activation_kwargs, norm_kwargs=norm_kwargs, conv_kwargs=conv_kwargs)
    block = torch.nn.Module()
    for element in elements:
        (name, module) = element
        block.add_module(name=name, module=module)
    return block


class ConvolutionBlock(torch.nn.Sequential):

    def __init__(self,
                 in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int]],
                 blockspec: str,
                 padding: Union[int, Tuple[int]],
                 padding_mode: str,
                 activation_kwargs: Optional[dict] = None,
                 norm_kwargs: Optional[dict] = None,
                 conv_kwargs: Optional[dict] = None) -> None:

        super(ConvolutionBlock, self).__init__()
        elements = create_conv_block_elements(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            blockspec=blockspec, padding=padding, padding_mode=padding_mode,
            activation_kwargs=activation_kwargs, norm_kwargs=norm_kwargs, conv_kwargs=conv_kwargs
        )
        for (name, module) in elements:
            self.add_module(name=name, module=module)
    



class Downsampler(torch.nn.Module):

    def __init__(self,
                 mode: str,
                 kernel_size: Union[int, Tuple[int]],
                 in_channels: Optional[Union[int, Tuple[int]]],
                 out_channels: Optional[Union[int, Tuple[int]]],
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 0):
        super(Downsampler, self).__init__()
        if mode == 'stridedconv':
            if in_channels is None or out_channels is None:
                raise TypeError('Downsampler via strided convolution requires in and out channel '
                                'specification')
            if stride == 1:
                warnings.warn('Using downsampling via strided convolution and stride = 1')
            
            self.downsample = torch.nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding    
            )
        elif mode in POOLING_MAPPING.keys():
            self.downsample = create_pooling(mode, kernel_size=kernel_size)

        else:
            message = (f'Invalid downsample mode: "{mode}"')
            raise ValueError(message)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)



class EncoderBlock(torch.nn.Module):

    def __init__(self,
                 mode: str,
                 in_channels: int,
                 out_channels: int,
                 blockspec: str,
                 convolution_kernel_size: Union[int, Tuple[int]],
                 downsample_kernel_size: Union[int, Tuple[int]],
                 downsample_stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 1,
                 padding_mode: str = 'zeros',
                 activation_kwargs: Optional[dict] = None,
                 norm_kwargs: Optional[dict] = None,
                 conv_kwargs: Optional[dict] = None) -> None:
        super(EncoderBlock, self).__init__()

        if mode in ('none', 'None', None):
            self.downsample_block = None
        else:
            # Here we do not change the feature maps on the downsampling operation.
            self.downsample_block = Downsampler(
                mode=mode, in_channels=in_channels, out_channels=in_channels,
                kernel_size=downsample_kernel_size, stride=downsample_stride
            )
        
        self.convolution_block = ConvolutionBlock(
            in_channels=in_channels, out_channels=out_channels,
            blockspec=blockspec, kernel_size=convolution_kernel_size,
            padding=padding, padding_mode=padding_mode,
            activation_kwargs=activation_kwargs, norm_kwargs=norm_kwargs,
            conv_kwargs=conv_kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample_block is not None:
            x = self.downsample_block(x)
        y = self.convolution_block(x)
        return y



class ModularCellNet(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 target_class_count: int,
                 fmapspec: Sequence[int],
                 blockspec: str,
                 final_pooling_mode: str,
                 final_pooling_out_size: Union[int, Tuple[int]],
                 conv_kernel_size: Union[int, Tuple[int]],
                 conv_padding: Union[int, Tuple[int]],
                 downsample_mode: str,
                 downsample_kernel_size: Union[int, Tuple[int]],
                 downsample_stride: Union[int, Tuple[int]],
                 conv_padding_mode: str = 'zeros',
                 activation_kwargs: Optional[dict] = None,
                 norm_kwargs: Optional[dict] = None,
                 conv_kwargs: Optional[dict] = None
                 ) -> None:

        super(ModularCellNet, self).__init__()

        self._apply_final_softmax: bool = False

        self.encoder_blocks = torch.nn.ModuleList()
        for i, out_channels in enumerate(fmapspec):
            if i == 0:
                mode = None
                current_in_channels = in_channels
            else:
                mode = downsample_mode
                current_in_channels = fmapspec[i - 1]
            
            block = EncoderBlock(
                mode=mode, in_channels=current_in_channels,
                out_channels=out_channels, blockspec=blockspec,
                convolution_kernel_size=conv_kernel_size,
                downsample_kernel_size=downsample_kernel_size,
                downsample_stride=downsample_stride,
                padding=conv_padding,
                padding_mode=conv_padding_mode, 
                activation_kwargs=activation_kwargs,
                norm_kwargs=norm_kwargs,
                conv_kwargs=conv_kwargs
            )
            self.encoder_blocks.append(block)

        flattened_size = np.prod((fmapspec[-1], *final_pooling_out_size))
        self.final_pooling = create_adaptive_pooling(
            final_pooling_mode, output_size=final_pooling_out_size
        )
        self.fc_layer = torch.nn.Linear(in_features=flattened_size,
                                        out_features=target_class_count)
        self.final_activation = torch.nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for encoder in self.encoder_blocks:
            x = encoder(x)
        x = self.final_pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layer(x)
        if self.apply_final_softmax:
            x = self.final_activation(x)
        return x

    @property
    def apply_final_softmax(self) -> bool:
        return self._apply_final_softmax

    @apply_final_softmax.setter
    def apply_final_softmax(self, new_state: bool) -> None:
        if not isinstance(new_state, bool):
            raise TypeError(
                f'attribute is restricted to boolean values, got {type(new_state)}'
            )
        self._apply_final_softmax = new_state




class LegacyCellNet(torch.nn.Module):
    """
    Emulate CNN by P. Fay in Anlauf-2.ipynb
    """

    def __init__(self,
                 N_classes: int,
                 conv_in_channels: int,
                 conv_out_channels: int,
                 kernel_size: Union[int, Tuple[int]],
                 nonlinearity: str,
                 nonlinearity_kwargs: Optional[dict] = None) -> None:

        super().__init__()
        nonlinearity_kwargs = nonlinearity_kwargs or {}
        self.nonlinearity = get_nonlinearity(nonlinearity, **nonlinearity_kwargs)

        self.norm = torch.nn.BatchNorm2d(num_features=1)

        self.convolution = torch.nn.Conv2d(
            in_channels=conv_in_channels,
            out_channels=conv_out_channels, kernel_size=kernel_size,
            padding='valid'
        )
        self.downsampling = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = torch.nn.Dropout2d(p=0.5)
        self.flatten = torch.nn.Flatten()
        self.linear_1 = torch.nn.Linear(in_features=219040, out_features=100)
        self.linear_2 = torch.nn.Linear(in_features=100, out_features=N_classes)
        self.final_activation = torch.nn.Softmax(dim=1)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.convolution(y)
        y = self.nonlinearity(y)
        y = self.downsampling(y)
        y = self.dropout(y)
        y = self.flatten(y)
        y = self.linear_1(y)
        y = self.nonlinearity(y)
        y = self.linear_2(y)
        return self.final_activation(y)





class CellNet(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        modules = OrderedDict([
            ('con1' , torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=2)),
            ('norm1' , torch.nn.BatchNorm2d(num_features=10)),
            ('nonlin1' , torch.nn.ReLU(inplace=True)),
            ('downsample1' , torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('con2' , torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=2)),
            ('norm2' , torch.nn.BatchNorm2d(num_features=20)),
            ('nonlin2' , torch.nn.ReLU(inplace=True)),
            ('downsample2' , torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('con3' , torch.nn.Conv2d(in_channels=20, out_channels=40, kernel_size=2)),
            ('norm3' , torch.nn.BatchNorm2d(num_features=40)),
            ('nonlin3' , torch.nn.ReLU(inplace=True)),
            ('downsample3' , torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('con4' , torch.nn.Conv2d(in_channels=40, out_channels=60, kernel_size=2)),
            ('norm4' , torch.nn.BatchNorm2d(num_features=60)),
            ('nonlin4' , torch.nn.ReLU(inplace=True)),
            ('downsample4' , torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('flatten', torch.nn.Flatten()),
            ('fc', torch.nn.Linear(in_features=17340, out_features=10))
        ])
        self.inner = torch.nn.Sequential(modules)
        
    def forward(self, x):
        return self.inner(x)


if __name__ == '__main__':
    t = torch.randn(size=(1, 1, 300, 300))
    model = CellNet()
    result = model(t)
    print(result.size())