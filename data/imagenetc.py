#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

import torch
import torch.multiprocessing
from torchvision import datasets
from torchvision import transforms

from args import args as args

torch.multiprocessing.set_sharing_strategy("file_system")


class ImageNetC:
    def __init__(self):
        super(ImageNetC, self).__init__()

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = (
            {"num_workers": args.workers, "pin_memory": True}
            if use_cuda
            else {}
        )

        # Data loading code
        valdir = os.path.join(args.data, "imagenet-c")
        valdir = f"{valdir}/{args.ct}/{args.sev}"

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.train_loader = None

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
