#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
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

    for i, model in enumerate(models):
        model.zero_grad()
        model.eval()

        if args.layerwise:
            for m in model.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                    Z = np.random.exponential(scale=1.0, size=args.n)
                    Z = Z / Z.sum()
                    for i in range(1, args.n):
                        setattr(m, f"t{i}", Z[i])
        else:
            Z = np.random.exponential(scale=1.0, size=args.n)
            Z = Z / Z.sum()
            for m in model.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                    for i in range(1, args.n):
                        setattr(m, f"t{i}", Z[i])

    test_loss = 0
    correct = 0
    corrects = [0 for _ in range(10)]

    M = 20
    acc_bm = torch.zeros(M)
    conf_bm = torch.zeros(M)
    count_bm = torch.zeros(M)

    val_loader = data_loader.val_loader

    for m in models:
        utils.update_bn(data_loader.train_loader, m, device=args.device)

    with torch.no_grad():

        for data, target in val_loader:
            data, target = data.to(args.device), target.to(args.device)

            model_output = models[0](data)
            model_pred = model_output.argmax(dim=1, keepdim=True)
            corrects[0] += (
                model_pred.eq(target.view_as(model_pred)).sum().item()
            )
            mean_output = model_output

            for i, m in enumerate(models[1:]):
                model_output = m(data)
                model_pred = model_output.argmax(dim=1, keepdim=True)
                corrects[i + 1] += (
                    model_pred.eq(target.view_as(model_pred)).sum().item()
                )
                mean_output += model_output

            mean_output /= len(models)
            # get the index of the max log-probability
            pred = mean_output.argmax(dim=1, keepdim=True)
            test_loss += criterion(mean_output, target).item()
            correct_vec = pred.eq(target.view_as(pred))
            correct += correct_vec.sum().item()
            soft_output = mean_output.softmax(dim=1)

            for i in range(data.size(0)):
                conf = soft_output[i][pred[i]]
                bin_idx = min((conf * M).int().item(), M - 1)
                acc_bm[bin_idx] += correct_vec[i].float().item()
                conf_bm[bin_idx] += conf.item()
                count_bm[bin_idx] += 1.0

    test_loss /= len(val_loader)
    test_acc = float(correct) / len(val_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({test_acc:.4f})\n"
    )

    ece = 0.0
    for i in range(M):
        ece += (acc_bm[i] - conf_bm[i]).abs().item()
    ece /= len(val_loader.dataset)
    print("ece is", ece)

    if args.save:
        writer.add_scalar(f"test/loss", test_loss, epoch)
        writer.add_scalar(f"test/acc", test_acc, epoch)

    corrects_sacled = [
        float(corrects[i]) / len(val_loader.dataset)
        for i in range(len(corrects))
    ]
    metrics = {
        f"model_{i}_acc": corrects_sacled[i]
        for i in range(len(corrects_sacled))
    }

    corrects_sacled = np.array(corrects_sacled)

    metrics["avg_model_acc"] = np.mean(corrects_sacled[corrects_sacled > 0])
    metrics["avg_model_std"] = np.std(corrects_sacled[corrects_sacled > 0])
    metrics["ece"] = ece
    metrics["test_loss"] = test_loss

    return test_acc, metrics
