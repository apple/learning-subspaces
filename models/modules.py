#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn
import torch.nn.functional as F

from args import args as pargs

StandardConv = nn.Conv2d
StandardBN = nn.BatchNorm2d


class SubspaceConv(nn.Conv2d):
    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w = self.get_weight()
        x = F.conv2d(
            x,
            w,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return x


class SubspaceBN(nn.BatchNorm2d):
    def forward(self, input):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w, b = self.get_weight()

        # The rest is code in the PyTorch source forward pass for batchnorm.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked
                    )
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (
                self.running_var is None
            )
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var
            if not self.training or self.track_running_stats
            else None,
            w,
            b,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class TwoParamConv(SubspaceConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.zeros_like(self.weight))

    def initialize(self, initialize_fn):
        initialize_fn(self.weight1)


class ThreeParamConv(SubspaceConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight2 = nn.Parameter(torch.zeros_like(self.weight))

    def initialize(self, initialize_fn):
        initialize_fn(self.weight1)
        initialize_fn(self.weight2)


class TwoParamBN(SubspaceBN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias1 = nn.Parameter(torch.Tensor(self.num_features))
        torch.nn.init.ones_(self.weight1)
        torch.nn.init.zeros_(self.bias1)


class ThreeParamBN(SubspaceBN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.weight1 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias1 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight2 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias2 = nn.Parameter(torch.Tensor(self.num_features))
        torch.nn.init.ones_(self.weight1)
        torch.nn.init.zeros_(self.bias1)
        torch.nn.init.ones_(self.weight2)
        torch.nn.init.zeros_(self.bias2)


class LinesConv(TwoParamConv):
    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight1
        return w


class LinesBN(TwoParamBN):
    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight1
        b = (1 - self.alpha) * self.bias + self.alpha * self.bias1
        return w, b


class CurvesConv(ThreeParamConv):
    def get_weight(self):
        w = (
            ((1 - self.alpha) ** 2) * self.weight
            + 2 * self.alpha * (1 - self.alpha) * self.weight2
            + (self.alpha ** 2) * self.weight1
        )
        return w


class CurvesBN(ThreeParamBN):
    def get_weight(self):
        w = (
            ((1 - self.alpha) ** 2) * self.weight
            + 2 * self.alpha * (1 - self.alpha) * self.weight2
            + (self.alpha ** 2) * self.weight1
        )
        b = (
            ((1 - self.alpha) ** 2) * self.bias
            + 2 * self.alpha * (1 - self.alpha) * self.bias2
            + (self.alpha ** 2) * self.bias1
        )
        return w, b
