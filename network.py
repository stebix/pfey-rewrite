import torch

from collections import OrderedDict
from typing import Union, Tuple, Optional

from utils import get_nonlinearity

import importlib

def get_norm(name: str, **kwargs) -> torch.nn.Module:
    module = importlib.import_module(name='torch.nn')
    norm_class = 0# getattr

    pass

def create_conv_block(in_channels: int, out_channels: int, kernel_size: Union[int, tuple]):
    conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
    norm = None #norm_class
    pass

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