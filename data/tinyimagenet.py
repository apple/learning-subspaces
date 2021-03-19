#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os

import numpy as np
import torch
import torch.multiprocessing
from torchvision import datasets
from torchvision import transforms

from args import args as args

torch.multiprocessing.set_sharing_strategy("file_system")


class TinyImageNet:
    def __init__(self):
        super(TinyImageNet, self).__init__()

        data_root = os.path.join(args.data, "tiny-imagenet-200")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = (
            {"num_workers": args.workers, "pin_memory": True}
            if use_cuda
            else {}
        )

        # Data loading code
        traindir = os.path.join(data_root, "train")
        valdir = os.path.join(data_root, "val")

        normalize = transforms.Normalize(
            mean=[0.480, 0.448, 0.397], std=[0.276, 0.269, 0.282]
        )

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomCrop(size=64, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

        self.val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                valdir, transforms.Compose([transforms.ToTensor(), normalize,]),
            ),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs,
        )
