import torch
import torch.nn as nn
from typing import Type, Union, List, Optional, Callable, Any
from utils import Conv  # Contains the Code to conv1x1 and conv3x3 functions

class BottleNeck(nn.Module):
    expansion: int = 4  # expansion factor for channels in bottleneck blocks

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        conv_size: int = 3,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        Initializes the BottleNeck module.

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

        width = out_channels * self.expansion

        # First 1x1 convolutional layer
        self.conv1 = Conv.conv1x1(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)

        # Second 3x3 convolutional layer
        self.conv2 = Conv.conv3x3(out_channels, out_channels, kernel_size=conv_size, stride=stride)
        self.bn2 = norm_layer(out_channels)

        # Third 1x1 convolutional layer
        self.conv3 = Conv.conv1x1(out_channels, width)
        self.bn3 = norm_layer(width)

        self.relu = nn.ReLU(inplace=True)
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
        Forward pass of the BottleNeck module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.relu(self.bn1(self.conv1(x)))  # First 1x1 convolution followed by batch norm and ReLU

        out = self.relu(self.bn2(self.conv2(out)))  # Second 3x3 convolution followed by batch norm and ReLU

        out = self.bn3(self.conv3(out))  # Third 1x1 convolution followed by batch norm

        # Residual connection
        out += self.skip(x)
        out = self.relu(out)  # Final ReLU activation

        return out
