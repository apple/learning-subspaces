#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
from models.cifar_resnet import CIFARResNet
from models.cifar_resnet_dropout import CIFARResNetDropout
from models.resnet import ResNet50
from models.resnet import WideResNet50_2
from models.tinyimagenetresnet import TinyImageNetResNet18
from models.tinyimagenetresnet import TinyImageNetResNet50

__all__ = [
    "CIFARResNet",
    "ResNet50",
    "WideResNet50_2",
    "CIFARResNetDropout",
    "TinyImageNetResNet18",
    "TinyImageNetResNet50",
]
