import torch
import torch.nn as nn
from typing import Type, Union, List, Optional, Callable, Any
from utils import conv3x3, conv1x1
class BottleNeck(nn.Module):
  expansion: int = 4

  def __init__(
      self, 
      in_channels: int,
      out_channels: int,
      stride: int = 1,
      conv_size: int = 3,
      norm_layer: Optional[Callable[..., nn.Module]] = None      
      ) -> None:
      super().__init__()
      if norm_layer is not None:
        norm_layer = nn.BatchNorm2d
      
      width = out_channels * self.expansion
      self.conv1 = conv1x1(in_channels, out_channels)
      self.bn1 = norm_layer(out_channels)
      self.conv2 = conv3x3(out_channels, out_channels, kernel_size=conv_size, stride=stride)
      self.bn2 = norm_layer(out_channels)
      self.conv = conv1x1(out_channels, width)
      self.bn3 = norm_layer(width)
      self.relu = nn.ReLU(inplace=True)
      self.stride = stride

      self.skip = nn.Sequential()
      if stride != 1 or in_channels != out_channels * self.expansion:
            self.skip = nn.Sequential(
                conv1x1(self.in_channels, out_channels * self.expansion, stride=stride),
                norm_layer(out_channels * self.expansion),
            )
  

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.relu(self.bn1(self.conv1(x)))

    out = self.relu(self.bn2(self.conv2(out)))

    out = self.bn3(self.conv3(out))
    
    out += self.skip(x)
    out = self.relu(out)

    return out

