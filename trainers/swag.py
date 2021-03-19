#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import math
import os

import numpy as np
import torch
import torch.nn as nn

import utils
from args import args


def init(models, writer, data_loader):
    return


def train(models, writer, data_loader, optimizers, criterion, epoch):
    return


def test(models, writer, criterion, data_loader, epoch):
    np.random.seed(args.seed)

    n = args.num_models
    # turn all the models into vectors
    vecs = [
        utils.sd_to_vector(models[i].state_dict()).clone() for i in range(n)
    ]

    swa_vec = vecs[0]
    for i in range(1, n):
        swa_vec = swa_vec + vecs[i]
    swa_vec = swa_vec / n

    square_vec = vecs[0].pow(2)
    for i in range(1, n):
        square_vec = square_vec + vecs[i].pow(2)
    square_vec = square_vec / n

    swa_diag_mult = (
        (1.0 / math.sqrt(2))
        * (square_vec - swa_vec.pow(2)).pow(0.5)
        * torch.randn_like(swa_vec)
    )

    low_rank_term = (vecs[0] - swa_vec) * torch.randn(1).item()
    for i in range(1, n):
        low_rank_term = (
            low_rank_term + (vecs[i] - swa_vec) * torch.randn(1).item()
        )
    low_rank_term = (1.0 / math.sqrt(2 * (n - 1))) * low_rank_term

    out = swa_vec + swa_diag_mult + low_rank_term

    final_model_sd = models[0].state_dict()
    utils.vector_to_sd(out, final_model_sd)
    models[0].load_state_dict(final_model_sd)

    utils.update_bn(data_loader.train_loader, models[0], device=args.device)

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
        os.path.join(args.tmp_dir, f"model_{args.j}.pt"),
    )

    test_acc = 0
    metrics = {}

    return test_acc, metrics
