#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn

import utils
from args import args


def init(models, writer, data_loader):
    return


def train(models, writer, data_loader, optimizers, criterion, epoch):
    return


def test(models, writer, criterion, data_loader, epoch):

    model = models[0]
    model_0 = models[1]
    model_0.eval()
    model_0.zero_grad()

    model.apply(lambda m: setattr(m, "return_feats", True))
    model_0.apply(lambda m: setattr(m, "return_feats", True))

    model.zero_grad()
    model.eval()
    test_loss = 0
    correct = 0
    ensemble_correct = 0
    m0_correct = 0
    tv_dist = 0.0
    val_loader = data_loader.val_loader
    feat_cosim = 0

    model.apply(lambda m: setattr(m, "alpha", args.alpha1))
    model_0.apply(lambda m: setattr(m, "alpha", args.alpha0))

    if args.update_bn:
        utils.update_bn(data_loader.train_loader, model, device=args.device)
        utils.update_bn(data_loader.train_loader, model_0, device=args.device)

    with torch.no_grad():

        for data, target in val_loader:
            data, target = data.to(args.device), target.to(args.device)

            output, feats = model(data)
            test_loss += criterion(output, target).item()

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

            # get model 0
            model_0_output, model_0_feats = model_0(data)
            ensemble_pred = (model_0_output + output).argmax(
                dim=1, keepdim=True
            )
            ensemble_correct += (
                ensemble_pred.eq(target.view_as(pred)).sum().item()
            )

            m0_pred = model_0_output.argmax(dim=1, keepdim=True)
            m0_correct += m0_pred.eq(target.view_as(pred)).sum().item()

            model_t_prob = nn.functional.softmax(output, dim=1)
            model_0_prob = nn.functional.softmax(model_0_output, dim=1)
            tv_dist += 0.5 * (model_0_prob - model_t_prob).abs().sum().item()

            feat_cosim += (
                torch.nn.functional.cosine_similarity(
                    feats, model_0_feats, dim=1
                )
                .pow(2)
                .sum()
                .item()
            )

    test_loss /= len(val_loader)
    test_acc = float(correct) / len(val_loader.dataset)
    m0_acc = float(m0_correct) / len(val_loader.dataset)
    tv_dist /= len(val_loader.dataset)
    feat_cosim /= len(val_loader.dataset)
    ensemble_acc = float(ensemble_correct) / len(val_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({test_acc:.4f})\n"
    )

    if args.save:
        writer.add_scalar(f"test/loss", test_loss, epoch)
        writer.add_scalar(f"test/acc", test_acc, epoch)

    metrics = {
        "tvdist": tv_dist,
        "ensemble_acc": ensemble_acc,
        "feat_cossim": feat_cosim,
        "m0_acc": m0_acc,
    }

    return test_acc, metrics
