#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import numpy as np
import torch
import torch.nn as nn

import utils
from args import args


def init(model, writer, train_loader):
    pass


def train(models, writer, data_loader, optimizers, criterion, epoch):
    print("Error -- train not implemented.")
    exit()


def test(models, writer, criterion, data_loader, epoch):

    for m in models:
        m.eval()
    test_loss = 0
    correct = 0
    val_loader = data_loader.val_loader

    for ms in zip(*[models[i].modules() for i in range(args.num_models)]):

        if isinstance(ms[0], nn.Conv2d):
            Z = np.random.exponential(
                1.0,
                size=(
                    args.num_models,
                    ms[0].weight.size(0),
                    ms[0].weight.size(1),
                    ms[0].weight.size(2),
                    ms[0].weight.size(3),
                ),
            )
            Zt = torch.from_numpy(Z).float().to(args.device)
            # normalize over the "num_models" dim
            Ztn = Zt / Zt.sum(0, keepdim=True)

            ms[0].weight.data = Ztn[0] * ms[0].weight.data
            for i in range(1, args.num_models):
                ms[0].weight.data += Ztn[i] * ms[i].weight.data
        elif isinstance(ms[0], nn.BatchNorm2d):
            Z = np.random.exponential(
                1.0, size=(args.num_models, ms[0].weight.size(0))
            )
            Zt = torch.from_numpy(Z).float().to(args.device)
            # normalize over the "num_models" dim
            Ztn = Zt / Zt.sum(0, keepdim=True)
            ms[0].weight.data = Ztn[0] * ms[0].weight.data
            for i in range(1, args.num_models):
                ms[0].weight.data += Ztn[i] * ms[i].weight.data

            Z = np.random.exponential(
                1.0, size=(args.num_models, ms[0].bias.size(0))
            )
            Zt = torch.from_numpy(Z).float().to(args.device)
            # normalize over the "num_models" dim
            Ztn = Zt / Zt.sum(0, keepdim=True)
            ms[0].bias.data = Ztn[0] * ms[0].bias.data
            for i in range(1, args.num_models):
                ms[0].bias.data += Ztn[i] * ms[i].bias.data

    model = models[0]
    utils.update_bn(data_loader.train_loader, model, device=args.device)

    model.eval()

    with torch.no_grad():

        for data, target in val_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += criterion(output, target).item()

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader)
    test_acc = float(correct) / len(val_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({test_acc:.4f})\n"
    )

    if args.save:
        writer.add_scalar(f"test/loss", test_loss, epoch)
        writer.add_scalar(f"test/acc", test_acc, epoch)

    metrics = {}

    return test_acc, metrics
