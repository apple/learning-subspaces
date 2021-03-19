#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import torch.nn as nn


def kaiming_normal(weight):
    nn.init.kaiming_normal_(weight,)
