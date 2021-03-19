#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os
import random
import sys

sys.path.append(os.path.abspath("."))
from args import args
from main import main as run


def samples(n, k):

    A = list(range(n))
    m = 2

    samples = set()

    if n == k:
        m = 1
    tries = 0
    while len(samples) < m:
        samples.add(tuple(sorted(random.sample(A, k))))
        tries += 1

    return samples


def main():
    train_epochs = 160
    args.label_noise = 0.0

    # TODO: change these paths -- this is an example.
    args.data = "~/data"
    args.log_dir = "learning-subspaces-results/tinyimagenet/eval-ensemble"

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    for num_models in [1, 2]:

        to_try = samples(2, num_models)

        for seed in range(len(to_try)):

            next_try = to_try.pop()

            args.seed = seed
            args.workers = 8
            args.output_size = 200
            args.set = "TinyImageNet"
            args.multigpu = [0]
            args.model = "TinyImageNetResNet18"
            args.conv_type = "StandardConv"
            args.bn_type = "StandardBN"
            args.conv_init = "kaiming_normal"
            args.trainer = "ensemble"
            args.epochs = 0
            args.resume = [
                f"learning-subspaces-results/tinyimagenet/train-ensemble-members/"
                f"id=base+ln={args.label_noise}+seed={c}"
                f"+try=0/epoch_{train_epochs}_iter_{train_epochs * 782}.pt"
                for c in next_try
            ]
            args.num_models = len(args.resume)
            args.name = (
                f"id=ensmeble+ln={args.label_noise}+epochs={train_epochs}"
                f"+num_models={args.num_models}+seed={seed}"
            )

            args.save = False
            args.save_data = True
            args.pretrained = True

            run()


if __name__ == "__main__":
    main()
