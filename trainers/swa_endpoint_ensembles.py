#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os

import torch
import torch.nn as nn

import utils
from args import args


def init(models, writer, data_loader):
    return


def train(models, writer, data_loader, optimizers, criterion, epoch):
    return


def test(models, writer, criterion, data_loader, epoch):
    j = args.j
    n = len(models)
    print(args.t)
    print((1 - args.t) / (n - 1))
    print("--")
    for ms in zip(*[model.modules() for model in models]):
        if isinstance(ms[0], nn.Conv2d):
            if j == 0:
                ms[0].weight.data = ms[0].weight.data * args.t
            else:
                ms[0].weight.data = ms[0].weight.data * (1 - args.t) / (n - 1)

            for i in range(1, n):
                if i == j:
                    ms[0].weight.data += ms[i].weight.data * args.t
                else:
                    ms[0].weight.data += (
                        ms[i].weight.data * (1 - args.t) / (n - 1)
                    )
            print("conv", ms[0].weight[0, 0, 0, 0])
        elif isinstance(ms[0], nn.BatchNorm2d):
            if j == 0:
                ms[0].weight.data = ms[0].weight.data * args.t
            else:
                ms[0].weight.data = ms[0].weight.data * (1 - args.t) / (n - 1)

            for i in range(1, n):
                if i == j:
                    ms[0].weight.data += ms[i].weight.data * args.t
                else:
                    ms[0].weight.data += (
                        ms[i].weight.data * (1 - args.t) / (n - 1)
                    )

            if j == 0:
                ms[0].bias.data = ms[0].bias.data * args.t
            else:
                ms[0].bias.data = ms[0].bias.data * (1 - args.t) / (n - 1)

            for i in range(1, n):
                if i == j:
                    ms[0].bias.data += ms[i].bias.data * args.t
                else:
                    ms[0].bias.data += ms[i].bias.data * (1 - args.t) / (n - 1)

            print("bn", ms[0].weight[0])
            print("bn", ms[0].bias[0])

    utils.update_bn(data_loader.train_loader, models[0], device=args.device)

    # here was save the model in args.tmp_dir/model_{j}.pt
    torch.save(
        {
            "epoch": 0,
            "iter": 0,
            "arch": args.model,
            "state_dicts": [models[0].state_dict()],
            "optimizers": None,
            "best_acc1": 0,
            "curr_acc1": 0,
        },
        os.path.join(args.tmp_dir, f"model_{j}.pt"),
    )

    test_acc = 0
    metrics = {}

    return test_acc, metrics
