#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import random

import numpy as np
import torch
import torch.nn as nn

import utils
from args import args


def get_weight(m, i):
    if i == 0:
        return m.weight
    return getattr(m, f"weight{i}")


def get_stats(model):
    norms = {}
    numerators = {}
    difs = {}
    cossim = 0
    l2 = 0
    nc2 = (args.n * (args.n - 1)) / 2.0
    for i in range(args.n):
        norms[f"{i}"] = 0.0
        for j in range(i + 1, args.n):
            numerators[f"{i}-{j}"] = 0.0
            difs[f"{i}-{j}"] = 0.0

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            for i in range(args.n):
                vi = get_weight(m, i)
                norms[f"{i}"] += vi.pow(2).sum()
                for j in range(i + 1, args.n):
                    vj = get_weight(m, j)
                    numerators[f"{i}-{j}"] += (vi * vj).sum()
                    difs[f"{i}-{j}"] += (vi - vj).pow(2).sum()

    for i in range(args.n):
        for j in range(i + 1, args.n):
            cossim += (1.0 / nc2) * (
                (
                    numerators[f"{i}-{j}"].pow(2)
                    / (norms[f"{i}"] * norms[f"{j}"])
                )
            )
            l2 += (1.0 / nc2) * difs[f"{i}-{j}"]

    l2 = l2.pow(0.5).item()
    cossim = cossim.item()
    return cossim, l2


def init(models, writer, data_loader):
    model = models[0]
    cossim, l2 = get_stats(model)
    if args.save:
        writer.add_scalar(f"test/norm", l2, -1)
        writer.add_scalar(f"test/cossim", cossim, -1)


def train(models, writer, data_loader, optimizers, criterion, epoch):

    model = models[0]
    optimizer = optimizers[0]

    model.zero_grad()
    model.train()
    avg_loss = 0.0
    train_loader = data_loader.train_loader

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)

        # To sample from a simplex, sample from an exponential distribution then renormalize.
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

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        if args.beta > 0:
            out = random.sample([i for i in range(args.n)], 2)
            i, j = out[0], out[1]
            num = 0.0
            normi = 0.0
            normj = 0.0
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    vi = get_weight(m, i)
                    vj = get_weight(m, j)
                    num += (vi * vj).sum()
                    normi += vi.pow(2).sum()
                    normj += vj.pow(2).sum()
            loss += args.beta * (num.pow(2) / (normi * normj))

        loss.backward()

        optimizer.step()

        avg_loss += loss.item()

        it = len(train_loader) * epoch + batch_idx
        if batch_idx % args.log_interval == 0:
            num_samples = batch_idx * len(data)
            num_epochs = len(train_loader.dataset)
            percent_complete = 100.0 * batch_idx / len(train_loader)
            print(
                f"Train Epoch: {epoch} [{num_samples}/{num_epochs} ({percent_complete:.0f}%)]\t"
                f"Loss: {loss.item():.6f}"
            )

            if args.save:
                writer.add_scalar(f"train/loss", loss.item(), it)
        if args.save and it in args.save_iters:
            utils.save_cpt(epoch, it, models, optimizers, -1, -1)

    avg_loss = avg_loss / len(train_loader)
    return avg_loss, optimizers


def test(models, writer, criterion, data_loader, epoch):

    model = models[0]
    model.eval()
    test_loss = 0
    correct0 = 0
    wa_correct = 0
    val_loader = data_loader.val_loader
    for i in range(1, args.n):
        model.apply(lambda m: setattr(m, f"t{i}", 1.0 / args.n))

    utils.update_bn(data_loader.train_loader, model, args.device)
    model.eval()
    cossim, l2 = get_stats(model)

    M = 20
    acc_bm = torch.zeros(M)
    conf_bm = torch.zeros(M)
    count_bm = torch.zeros(M)

    with torch.no_grad():

        for data, target in val_loader:
            data, target = data.to(args.device), target.to(args.device)

            wa_output = model(data)
            soft_output = wa_output.softmax(dim=1)
            wa_pred = wa_output.argmax(dim=1, keepdim=True)
            correct_vec = wa_pred.eq(target.view_as(wa_pred))
            wa_correct += correct_vec.sum().item()
            test_loss += criterion(wa_output, target).item()

            for i in range(data.size(0)):
                conf = soft_output[i][wa_pred[i]]
                bin_idx = min((conf * M).int().item(), M - 1)
                acc_bm[bin_idx] += correct_vec[i].float().item()
                conf_bm[bin_idx] += conf.item()
                count_bm[bin_idx] += 1.0

    wa_acc = float(wa_correct) / len(val_loader.dataset)
    m0_acc = float(correct0) / len(val_loader.dataset)
    test_acc = wa_acc
    test_loss /= len(val_loader)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({test_acc:.4f})\n"
    )

    if args.save:
        writer.add_scalar(f"test/loss", test_loss, epoch)
        writer.add_scalar(f"test/norm", l2, epoch)
        writer.add_scalar(f"test/cossim", cossim, epoch)

        writer.add_scalar(f"test/wa_acc", wa_acc, epoch)
        writer.add_scalar(f"test/m0_acc", m0_acc, epoch)

    ece = 0.0
    for i in range(M):
        ece += (acc_bm[i] - conf_bm[i]).abs().item()
    ece /= len(val_loader.dataset)
    print("ece is", ece)

    metrics = {
        "ece": ece,
        "wa_acc": wa_acc,
        "m0_acc": m0_acc,
        "l2": l2,
        "cossim": cossim,
        "test_loss": test_loss,
    }

    return test_acc, metrics
