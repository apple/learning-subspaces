#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os
import sys

sys.path.append(os.path.abspath("."))

from args import args
from main import main as run

if __name__ == "__main__":

    for seed in range(3):
        args.seed = seed
        args.label_noise = 0.0

        args.workers = 24
        args.wd = 0.00005
        args.batch_size = 256
        args.test_batch_size = 256
        args.output_size = 1000
        args.set = "ImageNet"
        args.multigpu = [0, 1, 2, 3]
        args.model = "WideResNet50_2"
        args.conv_type = "StandardConv"
        args.bn_type = "StandardBN"
        args.conv_init = "kaiming_normal"
        args.epochs = 200
        args.warmup_length = 5
        args.data_seed = 0
        args.name = f"id=base+ln={args.label_noise}+seed={args.seed}"
        args.save = True
        args.save_epochs = []
        args.save_iters = []

        # TODO: change these paths -- this is an example.
        args.data = "~/data"
        args.log_dir = (
            "learning-subspaces-results/imagenet/train-ensemble-members"
        )

        run()
