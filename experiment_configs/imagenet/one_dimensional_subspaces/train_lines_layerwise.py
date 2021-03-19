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
        args.lr = 0.1
        args.label_noise = 0.0
        args.beta = 1.0
        args.layerwise = True
        args.num_samples = 1

        args.test_freq = 100
        args.workers = 24
        args.wd = 0.00005
        args.batch_size = 256
        args.test_batch_size = 256
        args.output_size = 1000
        args.set = "ImageNet"
        args.multigpu = [0, 1, 2, 3]
        args.model = "WideResNet50_2"
        args.conv_type = "LinesConv"
        args.bn_type = "LinesBN"
        args.conv_init = "kaiming_normal"
        args.trainer = "train_one_dim_subspaces"
        args.epochs = 200
        args.warmup_length = 5
        args.data_seed = 0

        args.name = (
            f"id=lines-layerwise+ln={args.label_noise}"
            f"+beta={args.beta}"
            f"+num_samples={args.num_samples}"
            f"+seed={args.seed}"
        )

        args.save = True
        args.save_epochs = []
        args.save_iters = []

        # TODO: change these paths -- this is an example.
        args.data = "~/data"
        args.log_dir = (
            "learning-subspaces-results/imagenet/one-dimesnional-subspaces"
        )

        run()
