#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

# This file is modified from the open_lth repository: https://github.com/facebookresearch/open_lth
import torch.nn as nn
import torch.nn.functional as F

from args import args as pargs

from .builder import Builder


class CIFARResNet(nn.Module):
    """A residual neural network as originally designed for CIFAR-10."""

    class Block(nn.Module):
        """A ResNet block."""

        def __init__(self, f_in, f_out, downsample, builder, last_block):
            super(CIFARResNet.Block, self).__init__()
            stride = 2 if downsample else 1
            self.conv1 = builder.conv3x3(f_in, f_out, stride=stride)
            self.bn1 = builder.batchnorm(f_out)
            self.conv2 = builder.conv3x3(f_out, f_out)
            self.bn2 = builder.batchnorm(f_out)
            self.last_block = last_block
            self.return_feats = False

            # No parameters for shortcut connections.
            if downsample or f_in != f_out:
                self.shortcut = nn.Sequential(
                    builder.conv1x1(f_in, f_out, stride=2),
                    builder.batchnorm(f_out),
                )
            else:
                self.shortcut = nn.Sequential()

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            if self.last_block and self.return_feats:
                return F.relu(out), out
            return F.relu(out)

    def __init__(self):
        super(CIFARResNet, self).__init__()
        builder = Builder()

        # Previously in get model.
        name = pargs.model_name.split("_")
        W = 16 if len(name) == 3 else int(name[3])
        D = int(name[2])
        if (D - 2) % 3 != 0:
            raise ValueError("Invalid ResNet depth: {}".format(D))
        D = (D - 2) // 6
        plan = [(W, D), (2 * W, D), (4 * W, D)]

        outputs = pargs.output_size

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = builder.conv3x3(3, current_filters)
        self.bn = builder.batchnorm(current_filters)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(
                    CIFARResNet.Block(
                        current_filters,
                        filters,
                        downsample,
                        builder,
                        block_index == num_blocks - 1
                        and segment_index == len(plan) - 1,
                    )
                )
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)
        self.fc = builder.conv1x1(plan[-1][0], outputs)

        self.return_feats = False

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        if self.return_feats:
            out, feats = self.blocks(out)
            out = F.avg_pool2d(out, out.size()[3])
            out = self.fc(out)
            out = out.view(out.size(0), -1)
            return out, feats.view(feats.size(0), -1)
        else:
            out = self.blocks(out)
            out = F.avg_pool2d(out, out.size()[3])
            out = self.fc(out)
            out = out.view(out.size(0), -1)
            return out
