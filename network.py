from turtle import forward
import torch

from typing import Union, Sequence, Tuple, Optional

from utils import get_nonlinearity




class CellNet(torch.nn.Module):
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



if __name__ == '__main__':
    t = torch.randn(size=(1, 1, 300, 300))
    model = CellNet(N_classes=10, conv_in_channels=1, conv_out_channels=10,
                    kernel_size=4, nonlinearity='ReLU', nonlinearity_kwargs={'inplace' : True})
    result = model(t)
    print(result.size())