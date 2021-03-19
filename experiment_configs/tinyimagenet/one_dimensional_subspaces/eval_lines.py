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

    # TODO: change these paths -- this is an example.
    args.data = "~/data"
    args.log_dir = (
        "learning-subspaces-results/tinyimagenet/eval-one-dimesnional-subspaces"
    )

    for seed in range(2):
        args.epochs = 160
        args.seed = seed
        args.seed = 0
        args.lr = 0.1
        args.label_noise = 0.0
        args.beta = 1.0
        args.layerwise = False
        args.num_samples = 1

        args.test_freq = 10
        args.workers = 8
        args.output_size = 200
        args.set = "TinyImageNet"
        args.multigpu = [2]
        args.model = "TinyImageNetResNet18"
        args.conv_type = "LinesConv"
        args.bn_type = "LinesBN"
        args.conv_init = "kaiming_normal"

        name_string = (
            f"id=lines+ln={args.label_noise}"
            f"+beta={args.beta}"
            f"+num_samples={args.num_samples}"
            f"+seed={args.seed}"
        )

        # Now, analyze.

        args.resume = [
            f"learning-subspaces-results/tinyimagenet/one-dimesnional-subspaces/{name_string}+try=0/"
            f"epoch_{args.epochs}_iter_{args.epochs * 782}.pt"
        ] * 2

        args.num_models = 2
        args.save = False
        args.save_data = True
        args.pretrained = True
        args.epochs = 0
        args.trainer = "eval_one_dim_subspaces"
        args.update_bn = True

        acc_data = {}
        for i, alpha0 in enumerate(
            [
                0.0,
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                1.0,
            ]
        ):
            args.alpha0 = alpha0
            args.alpha1 = 1.0 - alpha0
            args.name = (
                f"{name_string}+alpha0={args.alpha0}+alpha1={args.alpha1}"
            )
            args.save_epochs = []
            run()
