# resnet.py
# 2019.07.24-Changed output of forward function
# Huawei Technologies Co., Ltd. <foss@huawei.com>
# Taken from https://github.com/huawei-noah/Data-Efficient-Model-Compression/blob/master/DAFL/resnet.py
# for comparison with DAFL

"""
This module implements ResNet models for image classification.
"""

from torch import nn
import torch.nn.functional as F
import torch
from typing import List, Type, Optional, Tuple


class BasicBlock(nn.Module):
    """
    A basic block for ResNet.
    """

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the BasicBlock.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    A bottleneck block for ResNet.
    """

    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Bottleneck block.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    A ResNet model.
    """

    def __init__(
        self,
        block: Type[nn.Module],
        num_blocks: List[int],
        num_classes: int = 10,
        num_channels: int = 3,
    ) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self, block: Type[nn.Module], planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, position: int = 0, out_feature: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the ResNet model.
        """
        if position == 0:
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)

        # print(x.shape) # [16, 64, 32, 32]
        if position <= 1:
            x = self.layer1(x)
        # print(x.shape) #  [16, 64, 32, 32]
        if position <= 2:
            x = self.layer2(x)
        # print(x.shape) # [16, 128, 16, 16]
        if position <= 3:
            x = self.layer3(x)
        # print(x.shape) # [16, 256, 8, 8]
        if position <= 4:
            x = self.layer4(x)
        # print(x.shape) # [16, 512, 4, 4]

        if position <= 5:
            x = F.avg_pool2d(x, 4)
            feature = x.view(x.size(0), -1)
            x = self.linear(feature)

            if out_feature:
                return x, feature
        return x


def resnet10(num_channels: int = 3, num_classes: int = 10) -> ResNet:
    """
    Constructs a ResNet-10 model.
    """
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes, num_channels)


def resnet18(num_channels: int = 3, num_classes: int = 10) -> ResNet:
    """
    Constructs a ResNet-18 model.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, num_channels)


def resnet34(num_channels: int = 3, num_classes: int = 10) -> ResNet:
    """
    Constructs a ResNet-34 model.
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, num_channels)


def resnet50(num_channels: int = 3, num_classes: int = 10) -> ResNet:
    """
    Constructs a ResNet-50 model.
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, num_channels)


def resnet101(num_channels: int = 3, num_classes: int = 10) -> ResNet:
    """
    Constructs a ResNet-101 model.
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, num_channels)


def resnet152(num_channels: int = 3, num_classes: int = 10) -> ResNet:
    """
    Constructs a ResNet-152 model.
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, num_channels)


# model=ResNet34()
# img=torch.randn((1,3,32,32))
# print(model.forward(img,0))
