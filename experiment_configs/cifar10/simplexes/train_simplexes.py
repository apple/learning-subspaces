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

    for seed in range(2):
        args.seed = seed
        args.lr = 0.1
        args.label_noise = 0.0
        args.beta = 1.0
        args.layerwise = False
        args.num_samples = 1

        # n is the number of endpoints of the simplex
        for n in [2, 3, 4, 5, 6]:
            args.n = n

            args.test_freq = 10
            args.set = "CIFAR10"
            args.multigpu = [0]
            args.model = "CIFARResNet"
            args.model_name = "cifar_resnet_20"
            args.conv_type = f"SimplexConv{n}"
            args.bn_type = f"SimplexBN{n}"
            args.conv_init = "kaiming_normal"
            args.trainer = "train_simplexes"
            args.epochs = 160
            args.warmup_length = 5
            args.data_seed = 0
            args.train_update_bn = True
            args.update_bn = True

            args.name = (
                f"id=simplex+n={n}+ln={args.label_noise}"
                f"+beta={args.beta}"
                f"+num_samples={args.num_samples}"
                f"+seed={args.seed}"
            )

            args.save = True
            args.save_epochs = []
            args.save_iters = []

            # TODO: change these paths -- this is an example.
            args.data = "~/data"
            args.log_dir = "learning-subspaces-results/cifar/simplexes"

            run()
