import torch
import torch.nn as nn
from typing import Type, Union, List, Optional, Callable, Any
from utils import Conv # Contains the Code to conv1x1 and conv3x3 functions

class BasicBlock(nn.Module):
    expansion: int = 1  # expansion factor for channels in residual connections

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        conv_size: int = 3,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        Initializes the BasicBlock module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride of the first convolutional layer. Default is 1.
            conv_size (int): Size of the convolutional kernel. Default is 3.
            norm_layer (Optional[Callable[..., nn.Module]]): Optional normalization layer. Default is None.
        """
        super().__init__()
        if norm_layer is not None:
            norm_layer = nn.BatchNorm2d

        # First convolutional layer
        self.conv1 = Conv.conv3x3(in_channels, out_channels, conv_size, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = Conv.conv3x3(in_channels=out_channels, out_channels=out_channels)
        self.bn2 = norm_layer(out_channels)
        self.stride = stride

        # Skip connection (residual connection)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.skip = nn.Sequential(
                Conv.conv1x1(in_channels, out_channels * self.expansion, stride=stride),
                norm_layer(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BasicBlock module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection
        out += self.skip(x)
        out = self.relu(out)

        return out
