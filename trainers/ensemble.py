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
    print("Error -- train not implemented. Only use this for testing.")
    exit()


def test(models, writer, criterion, data_loader, epoch):

    for model in models:
        model.eval()
        model.apply(lambda m: setattr(m, "return_feats", True))
    test_loss = 0
    correct = 0
    tvdist_sum = 0
    tvdist_len = 0
    feat_cossim = 0
    percent_disagree_sum = 0
    percent_disagree_len = 0
    percent_disagree_correct_sum = 0
    percent_disagree_correct_len = 0
    val_loader = data_loader.val_loader

    if args.update_bn:
        for model in models:
            utils.update_bn(data_loader.train_loader, model, args.device)

    with torch.no_grad():

        for data, target in val_loader:
            data, target = data.to(args.device), target.to(args.device)

            output, f = models[0](data)
            probs = [nn.functional.softmax(output, dim=1)]
            feats = [f]
            for t in range(1, args.num_models):
                modelt_output, model_feats_t = models[t](data)
                feats.append(model_feats_t)
                probs.append(nn.functional.softmax(modelt_output, dim=1))
                output += modelt_output

            # output = 0
            # for p in probs:
            #     output += p.log()
            # output = (output / args.num_models).exp()

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

            # get tvdist between i and j
            for i in range(args.num_models):
                for j in range(i + 1, args.num_models):
                    feat_cossim += (
                        nn.functional.cosine_similarity(
                            feats[i], feats[j], dim=1
                        )
                        .sum()
                        .item()
                    )
                    pairwise_tvdist = 0.5 * (probs[i] - probs[j]).abs().sum(
                        dim=1
                    )
                    tvdist_len += pairwise_tvdist.size(0)
                    tvdist_sum += pairwise_tvdist.sum().item()

                    model_i_pred = probs[i].argmax(dim=1, keepdim=True)
                    model_j_pred = probs[j].argmax(dim=1, keepdim=True)
                    percent_disagree_len += data.size(0)
                    percent_disagree_sum += (
                        (model_i_pred != model_j_pred).sum().item()
                    )

                    percent_disagree_correct_len += data.size(0)
                    percent_disagree_correct_sum += (
                        (
                            (model_i_pred != model_j_pred)
                            * (
                                model_i_pred.eq(target.view_as(model_i_pred))
                                + model_j_pred.eq(target.view_as(model_j_pred))
                            )
                        )
                        .sum()
                        .item()
                    )

    feat_cossim = feat_cossim / tvdist_len if tvdist_len > 0 else 0
    tvdist = tvdist_sum / tvdist_len if tvdist_len > 0 else 0
    percent_disagree = (
        percent_disagree_sum / percent_disagree_len
        if percent_disagree_len > 0
        else 0
    )
    percent_disagree_correct = (
        percent_disagree_correct_sum / percent_disagree_correct_len
        if percent_disagree_correct_len > 0
        else 0
    )
    test_loss /= len(val_loader)
    test_acc = float(correct) / len(val_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({test_acc:.4f}), TVDist: ({tvdist})\n"
    )

    if args.save:
        writer.add_scalar(f"test/loss", test_loss, epoch)
        writer.add_scalar(f"test/acc", test_acc, epoch)

    metrics = {
        "tvdist": tvdist,
        "percent_disagree": percent_disagree,
        "percent_disagree_correct": percent_disagree_correct,
        "feat_cossim": feat_cossim,
    }

    return test_acc, metrics
