#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import os

import numpy as np
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from args import args


class CIFAR10:
    def __init__(self):
        super(CIFAR10, self).__init__()

        data_root = os.path.join(args.data, "cifar10")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = (
            {"num_workers": args.workers, "pin_memory": True}
            if use_cuda
            else {}
        )

        # mirrors open_lth: https://github.com/facebookresearch/open_lth
        normalize = torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )

        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        if args.label_noise is not None:
            print(f"==> Using label noising proportion {args.label_noise}")

            pfile = "cifar"
            n = len(train_dataset.data)
            if not os.path.isfile(pfile + ".npy"):
                perm = np.random.permutation(n)
                labels = np.random.randint(10, size=(n,))
                np.save(pfile, perm)
                np.save(pfile + "_labels", labels)
            else:
                perm = np.load(pfile + ".npy")
                labels = np.load(pfile + "_labels.npy")

            for k in range(int(n * args.label_noise)):
                train_dataset.targets[perm[k]] = labels[k]

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
        )
