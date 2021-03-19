#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
from args import args

from . import init
from . import modules
from . import modules_gen


class Builder(object):
    def __init__(self):
        if hasattr(modules, args.conv_type):
            self.conv_layer = getattr(modules, args.conv_type)
        else:
            self.conv_layer = getattr(modules_gen, args.conv_type)

        if hasattr(modules, args.bn_type):
            self.bn_layer = getattr(modules, args.bn_type)
        else:
            self.bn_layer = getattr(modules_gen, args.bn_type)

        self.conv_init = getattr(init, args.conv_init)

    def conv(
        self,
        kernel_size,
        in_planes,
        out_planes,
        stride=1,
        groups=1,
        first_layer=False,
        last_layer=False,
        is_conv=False,
    ):

        if kernel_size == 1:
            conv = self.conv_layer(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=False
            )
        elif kernel_size == 3:
            conv = self.conv_layer(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=False,
            )
        elif kernel_size == 5:
            conv = self.conv_layer(
                in_planes,
                out_planes,
                kernel_size=5,
                stride=stride,
                padding=2,
                groups=groups,
                bias=False,
            )
        elif kernel_size == 7:
            conv = self.conv_layer(
                in_planes,
                out_planes,
                kernel_size=7,
                stride=stride,
                padding=3,
                groups=groups,
                bias=False,
            )
        else:
            return None

        conv.first_layer = first_layer
        conv.last_layer = last_layer
        conv.is_conv = is_conv
        self.conv_init(conv.weight)
        if hasattr(conv, "initialize"):
            conv.initialize(self.conv_init)
        return conv

    def conv1x1(
        self,
        in_planes,
        out_planes,
        stride=1,
        groups=1,
        first_layer=False,
        last_layer=False,
        is_conv=False,
    ):
        """1x1 convolution with padding"""
        c = self.conv(
            1,
            in_planes,
            out_planes,
            stride=stride,
            groups=groups,
            first_layer=first_layer,
            last_layer=last_layer,
            is_conv=is_conv,
        )

        return c

    def conv3x3(
        self,
        in_planes,
        out_planes,
        stride=1,
        groups=1,
        first_layer=False,
        last_layer=False,
        is_conv=False,
    ):
        """3x3 convolution with padding"""
        c = self.conv(
            3,
            in_planes,
            out_planes,
            stride=stride,
            groups=groups,
            first_layer=first_layer,
            last_layer=last_layer,
            is_conv=is_conv,
        )
        return c

    def conv5x5(
        self,
        in_planes,
        out_planes,
        stride=1,
        groups=1,
        first_layer=False,
        last_layer=False,
        is_conv=False,
    ):
        """5x5 convolution with padding"""
        c = self.conv(
            5,
            in_planes,
            out_planes,
            stride=stride,
            groups=groups,
            first_layer=first_layer,
            last_layer=last_layer,
            is_conv=is_conv,
        )
        return c

    def conv7x7(
        self,
        in_planes,
        out_planes,
        stride=1,
        groups=1,
        first_layer=False,
        last_layer=False,
        is_conv=False,
        layer=-1,
    ):
        """7x7 convolution with padding"""
        c = self.conv(
            7,
            in_planes,
            out_planes,
            stride=stride,
            groups=groups,
            first_layer=first_layer,
            last_layer=last_layer,
            is_conv=is_conv,
        )
        return c

    def batchnorm(self, planes):
        return self.bn_layer(planes)
