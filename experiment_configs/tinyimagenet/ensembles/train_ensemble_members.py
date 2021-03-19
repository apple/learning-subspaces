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

        args.workers = 8
        args.output_size = 200
        args.set = "TinyImageNet"
        args.multigpu = [0]
        args.model = "TinyImageNetResNet18"
        args.conv_type = "StandardConv"
        args.bn_type = "StandardBN"
        args.conv_init = "kaiming_normal"
        args.epochs = 160
        args.warmup_length = 5
        args.data_seed = 0
        args.name = f"id=base+ln={args.label_noise}+seed={args.seed}"
        args.save = True
        args.save_epochs = []
        args.save_iters = []

        # TODO: change these paths -- this is an example.
        args.data = "~/data"
        args.log_dir = (
            "learning-subspaces-results/tinyimagenet/train-ensemble-members"
        )

        run()
