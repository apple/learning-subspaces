#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
# Modified from https://github.com/ganguli-lab/Synaptic-Flow
import torch.nn as nn

from args import args as pargs

from .builder import Builder


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34"""

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        base_width=64,
        builder=None,
        last_block=False,
    ):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            builder.conv3x3(in_channels, out_channels, stride=stride),
            builder.batchnorm(out_channels),
            nn.ReLU(inplace=True),
            builder.conv3x3(out_channels, out_channels * BasicBlock.expansion),
            builder.batchnorm(out_channels * BasicBlock.expansion),
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                builder.conv1x1(
                    in_channels,
                    out_channels * BasicBlock.expansion,
                    stride=stride,
                ),
                builder.batchnorm(out_channels * BasicBlock.expansion),
            )
        self.return_feats = False
        self.last_block = last_block

    def forward(self, x):
        fts = self.residual_function(x) + self.shortcut(x)
        out = nn.ReLU(inplace=True)(fts)
        if self.last_block and self.return_feats:
            return out, fts
        return out


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers"""

    expansion = 4

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        base_width=64,
        builder=None,
        last_block=False,
    ):
        super().__init__()
        width = int(out_channels * (base_width / 64.0))
        self.residual_function = nn.Sequential(
            builder.conv1x1(in_channels, width),
            builder.batchnorm(width),
            nn.ReLU(inplace=True),
            builder.conv3x3(width, width, stride=stride),
            builder.batchnorm(width),
            nn.ReLU(inplace=True),
            builder.conv1x1(width, out_channels * BottleNeck.expansion),
            builder.batchnorm(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                builder.conv1x1(
                    in_channels,
                    out_channels * BottleNeck.expansion,
                    stride=stride,
                ),
                builder.batchnorm(out_channels * BottleNeck.expansion),
            )
        self.return_feats = False
        self.last_block = last_block

    def forward(self, x):
        fts = self.residual_function(x) + self.shortcut(x)
        out = nn.ReLU(inplace=True)(fts)
        if self.last_block and self.return_feats:
            return out, fts
        return out


class TinyImageNetResNet(nn.Module):
    def __init__(self, block, num_block, base_width, num_classes=200):
        super().__init__()
        builder = Builder()
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            builder.conv3x3(3, 64), builder.batchnorm(64), nn.ReLU(inplace=True)
        )
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(
            block, 64, num_block[0], 1, base_width, builder
        )
        self.conv3_x = self._make_layer(
            block, 128, num_block[1], 2, base_width, builder
        )
        self.conv4_x = self._make_layer(
            block, 256, num_block[2], 2, base_width, builder
        )
        self.conv5_x = self._make_layer(
            block, 512, num_block[3], 2, base_width, builder
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = builder.conv1x1(512 * block.expansion, num_classes)
        self.return_feats = False

    def _make_layer(
        self, block, out_channels, num_blocks, stride, base_width, builder
    ):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layer_list = []
        for i, stride in enumerate(strides):
            layer_list.append(
                block(
                    self.in_channels,
                    out_channels,
                    stride,
                    base_width,
                    builder,
                    i == (len(strides) - 1) and out_channels == 512,
                )
            )
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layer_list)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        if self.return_feats:
            output, fts = self.conv5_x(output)
        else:
            output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = self.fc(output)

        if self.return_feats:
            return output.squeeze(), fts.view(fts.size(0), -1)
        return output.squeeze()


def _resnet(arch, block, num_block, base_width):
    model = TinyImageNetResNet(block, num_block, base_width, pargs.output_size)
    return model


def TinyImageNetResNet18():
    """return a ResNet 18 object"""
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], 64)


def TinyImageNetResNet50():
    """return a ResNet 18 object"""
    return _resnet("resnet50", BottleNeck, [3, 4, 6, 3], 64)
