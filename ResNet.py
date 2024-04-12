from typing import Type, Union, List, Optional, Callable, Any
from utils import Conv  # Import utility functions
from BasicBlock import *  # Import BasicBlock class
from BottleNeck import *  # Import BottleNeck class

class ResNet(nn.Module):
    def __init__(
        self, 
        block: List[Type[Union[BasicBlock, BottleNeck]]],
        num_blocks: List[int],
        channel_size: List[int],
        conv_kernel_size: int,
        num_classes: int = 10,
        zero_init_residual: bool = False,
        he_init: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = None   
    ) -> None:
        """
        ResNet constructor.

        Args:
            block (List[Type[Union[BasicBlock, BottleNeck]]]): List of block types used in ResNet.
            num_blocks (List[int]): List of numbers of blocks in each layer.
            channel_size (List[int]): List of channel sizes for each layer.
            conv_kernel_size (int): Kernel size for the initial convolutional layer.
            num_classes (int): Number of output classes (default is 10 for CIFAR10).
            zero_init_residual (bool): If True, initialize residual connections to zero (default is False).
            he_init (bool): If True, use He initialization for convolutional layers (default is False).
            norm_layer (Optional[Callable[..., nn.Module]]): Normalization layer to use (default is BatchNorm2d).
        """
        super().__init__()
        # Default to BatchNorm2d if no norm_layer is provided
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.norm_layer = norm_layer

        # Initial convolution and normalization
        self.in_channels = channel_size[0]
        self.conv_kernel_size = conv_kernel_size

        self.conv1 = nn.Conv2d(3, channel_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(channel_size[0])
        self.relu = nn.ReLU(inplace=True)

        # Building ResNet layers using specified block types and channel sizes
        self.layer1 = self._make_layer_(block[0], channel_size[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer_(block[1], channel_size[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer_(block[2], channel_size[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer_(block[3], channel_size[3], num_blocks[3], stride=2)
        
        # Adaptive pooling and classifier layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel_size[3] * block[3].expansion, num_classes)

        # Initialize weights using He initialization for convolutional layers
        if he_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  

    def _make_layer_(
        self,
        block: Type[Union[BasicBlock, BottleNeck]],
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """
        Create a layer of blocks for the network.

        Args:
            block (Type[Union[BasicBlock, BottleNeck]]): Type of block to use.
            out_channels (int): Number of output channels for the layer.
            num_blocks (int): Number of blocks in the layer.
            stride (int): Stride for the layer.

        Returns:
            nn.Sequential: Sequential container of blocks.
        """
        norm_layer = self.norm_layer
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, self.conv_kernel_size, norm_layer))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
