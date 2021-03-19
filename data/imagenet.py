#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch
import torch.multiprocessing
from torchvision import datasets
from torchvision import transforms

from args import args as args

torch.multiprocessing.set_sharing_strategy("file_system")


class ImageNet:
    def __init__(self):
        super(ImageNet, self).__init__()

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = (
            {"num_workers": args.workers, "pin_memory": True}
            if use_cuda
            else {}
        )

        data_root = os.path.join(args.data, "imagenet")
        traindir = os.path.join(data_root, "train")
        valdir = os.path.join(data_root, "val_in_folder")

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        if args.label_noise is not None:
            print(f"==> Using label noising proportion {args.label_noise}")

            pfile = "imagenet"
            n = len(train_dataset.targets)  # 1281167

            if not os.path.isfile(pfile + ".npy"):
                perm = np.random.permutation(n)
                labels = np.random.randint(1000, size=(n,))
                np.save(pfile, perm)
                np.save(pfile + "_labels", labels)
            else:
                perm = np.load(pfile + ".npy")
                labels = np.load(pfile + "_labels.npy")

            for k in range(int(n * args.label_noise)):
                train_dataset.samples[perm[k]] = (
                    train_dataset.samples[perm[k]][0],
                    labels[k],
                )
            train_dataset.targets = [s[1] for s in train_dataset.samples]

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

        self.val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                valdir,
                transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs,
        )
