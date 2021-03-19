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
from args import args as pargs
from models.modules import ThreeParamConv
from models.modules import TwoParamBN
from models.modules import TwoParamConv


def get_stats(model):
    norms = {}
    numerators = {}
    difs = {}
    cossim = 0
    l2 = 0
    num_points = 2 if args.conv_type is "LinesConv" else 3

    for i in range(num_points):
        norms[f"{i}"] = 0.0
        for j in range(i + 1, num_points):
            numerators[f"{i}-{j}"] = 0.0
            difs[f"{i}-{j}"] = 0.0

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            for i in range(num_points):
                vi = get_weight(m, i)
                norms[f"{i}"] += vi.pow(2).sum()
                for j in range(i + 1, num_points):
                    vj = get_weight(m, j)
                    numerators[f"{i}-{j}"] += (vi * vj).sum()
                    difs[f"{i}-{j}"] += (vi - vj).pow(2).sum()

    for i in range(num_points):
        for j in range(i + 1, num_points):
            cossim += numerators[f"{i}-{j}"].pow(2) / (
                norms[f"{i}"] * norms[f"{j}"]
            )
            l2 += difs[f"{i}-{j}"]

    l2 = l2.pow(0.5).item()
    cossim = cossim.item()
    return cossim, l2


def get_weight(m, i):
    if i == 0:
        return m.weight
    return getattr(m, f"weight{i}")


def init(models, writer, data_loader):
    pass


def train(models, writer, data_loader, optimizers, criterion, epoch):

    # We consider only a single model here. Multiple models are for ensembles and SWA baselines.
    model = models[0]
    optimizer = optimizers[0]

    if args.num_samples > 1:
        model.apply(lambda m: setattr(m, "return_feats", True))

    model.zero_grad()
    model.train()
    avg_loss = 0.0
    train_loader = data_loader.train_loader

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)

        # num_samples is the number of samples to draw from the subspace for batch.
        # in all experiments in the main paper it is 1.
        if args.num_samples == 1:

            if args.layerwise:
                for m in model.modules():
                    if isinstance(m, nn.Conv2d) or isinstance(
                        m, nn.BatchNorm2d
                    ):
                        alpha = np.random.uniform(0, 1)
                        setattr(m, f"alpha", alpha)
            else:
                alpha = np.random.uniform(0, 1)
                for m in model.modules():
                    if isinstance(m, nn.Conv2d) or isinstance(
                        m, nn.BatchNorm2d
                    ):
                        setattr(m, f"alpha", alpha)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

        else:
            # Feel free to ignore this part as it is often not used.
            # This corresponds to section B of the appendix, where multiple samples from the subsapce are
            # taken for each batch.
            div = data.size(0) // args.num_samples
            feats = []
            ts = []
            optimizer.zero_grad()

            for sample in range(args.num_samples):

                if args.layerwise:
                    for m in model.modules():
                        if isinstance(m, nn.Conv2d) or isinstance(
                            m, nn.BatchNorm2d
                        ):
                            alpha = np.random.uniform(0, 1)
                            setattr(m, f"alpha", alpha)
                else:
                    alpha = np.random.uniform(0, 1)
                    for m in model.modules():
                        if isinstance(m, nn.Conv2d) or isinstance(
                            m, nn.BatchNorm2d
                        ):
                            setattr(m, f"alpha", alpha)

                output, f = model(data[sample * div : (sample + 1) * div])
                feats.append(f)

                if sample == 0:
                    loss = (
                        criterion(
                            output, target[sample * div : (sample + 1) * div]
                        )
                        / args.num_samples
                    )
                else:
                    loss += (
                        criterion(
                            output, target[sample * div : (sample + 1) * div]
                        )
                        / args.num_samples
                    )

            if args.lamb > 0:
                out = random.sample([i for i in range(args.num_samples)], 2)
                i, j = out[0], out[1]
                fi, fj = feats[i], feats[j]
                ti, tj = ts[i], ts[j]
                loss += (
                    args.fcos_weight
                    * abs(ti - tj)
                    * (
                        (fi * fj).sum().pow(2)
                        / (fi.pow(2).sum() * fj.pow(2).sum())
                    )
                )

        # Application of the regularization term, equation 3.
        num_points = 2 if args.conv_type is "LinesConv" else 3
        if args.beta > 0:
            out = random.sample([i for i in range(num_points)], 2)

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

    model.apply(lambda m: setattr(m, "return_feats", False))

    avg_loss = avg_loss / len(train_loader)
    return avg_loss, optimizers


def test(models, writer, criterion, data_loader, epoch):
    model = models[0]

    model.zero_grad()
    model.eval()
    test_loss = 0
    correct = 0
    val_loader = data_loader.val_loader

    model.apply(lambda m: setattr(m, "alpha", 0.5))

    # optionally update the bn during training to, but note this slows down things.
    if args.train_update_bn:
        utils.update_bn(data_loader.train_loader, model, args.device)

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
    cossim, l2 = get_stats(model)

    if args.save:
        writer.add_scalar(f"test/loss", test_loss, epoch)
        writer.add_scalar(f"test/acc", test_acc, epoch)
        writer.add_scalar(f"test/norm", l2, epoch)
        writer.add_scalar(f"test/cossim", cossim, epoch)

    metrics = {"norm": l2, "cossim": cossim}

    return test_acc, metrics
