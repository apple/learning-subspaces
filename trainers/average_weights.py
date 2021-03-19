#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
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
    model = models[0]
    test_loss = 0
    correct = 0
    val_loader = data_loader.val_loader

    M = 20
    acc_bm = torch.zeros(M)
    conf_bm = torch.zeros(M)
    count_bm = torch.zeros(M)

    for ms in zip(*[models[i].modules() for i in range(args.num_models)]):
        if isinstance(ms[0], nn.Conv2d):
            ms[0].weight.data = (1.0 / args.num_models) * ms[0].weight.data
            for i in range(1, args.num_models):
                ms[0].weight.data += (1.0 / args.num_models) * ms[i].weight.data
        elif isinstance(ms[0], nn.BatchNorm2d):
            ms[0].weight.data = (1.0 / args.num_models) * ms[0].weight.data
            for i in range(1, args.num_models):
                ms[0].weight.data += (1.0 / args.num_models) * ms[i].weight.data
            ms[0].bias.data = (1.0 / args.num_models) * ms[0].bias.data
            for i in range(1, args.num_models):
                ms[0].bias.data += (1.0 / args.num_models) * ms[i].bias.data

    utils.update_bn(data_loader.train_loader, model, device=args.device)

    with torch.no_grad():

        for data, target in val_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += criterion(output, target).item()

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            soft_out = output.softmax(dim=1)
            correct_vec = pred.eq(target.view_as(pred))

            correct += correct_vec.sum().item()

            for i in range(data.size(0)):
                conf = soft_out[i][pred[i]]
                bin_idx = min((conf * M).int().item(), M - 1)
                acc_bm[bin_idx] += correct_vec[i].float().item()
                conf_bm[bin_idx] += conf.item()
                count_bm[bin_idx] += 1.0

    test_loss /= len(val_loader)
    test_acc = float(correct) / len(val_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({test_acc:.4f})\n"
    )

    if args.save:
        writer.add_scalar(f"test/loss", test_loss, epoch)
        writer.add_scalar(f"test/acc", test_acc, epoch)

    ece = 0.0
    for i in range(M):
        ece += (acc_bm[i] - conf_bm[i]).abs().item()
    ece /= len(val_loader.dataset)
    print("ece is", ece)

    metrics = {"ece": ece, "test_loss": test_loss}

    return test_acc, metrics
