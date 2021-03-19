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
        "learning-subspaces-results/imagenet/eval-one-dimesnional-subspaces"
    )

    args.seed = 0
    args.lr = 0.1
    args.label_noise = 0.0
    args.beta = 1.0
    args.layerwise = False
    args.num_samples = 1

    args.test_freq = 10
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
    args.epochs = 200

    name_string = (
        f"id=lines+ln={args.label_noise}"
        f"+beta={args.beta}"
        f"+num_samples={args.num_samples}"
        f"+seed={args.seed}"
    )

    # Now, analyze.

    args.resume = [
        f"learning-subspaces-results/imagenet/one-dimesnional-subspaces/{name_string}+try=0/"
        f"epoch_{args.epochs}_iter_{args.epochs * 5005}.pt"
    ] * 2

    args.num_models = 2
    args.save = False
    args.save_data = True
    args.pretrained = True
    args.epochs = 1
    args.trainer = "eval_one_dim_subspaces_multigpu"
    args.update_bn = False

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
        args.name = f"{name_string}+alpha0={args.alpha0}+alpha1={args.alpha1}"
        args.save_epochs = []
        run()
