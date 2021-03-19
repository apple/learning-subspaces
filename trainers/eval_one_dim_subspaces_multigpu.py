#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import time

import torch
import torch.nn as nn

import utils
from args import args


def init(models, writer, data_loader):
    return


def train(models, writer, data_loader, optimizers, criterion, epoch):

    for model in models:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()

    model = models[0]
    model_0 = models[1]
    optimizer = optimizers[0]

    model.zero_grad()
    model_0.zero_grad()
    model.train()
    model_0.train()
    avg_loss = 0.0
    train_loader = data_loader.train_loader

    model.apply(lambda m: setattr(m, "alpha", args.alpha1))
    model_0.apply(lambda m: setattr(m, "alpha", args.alpha0))

    with torch.no_grad():
        end = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            data_time = time.time() - end

            optimizer.zero_grad()

            model(data)
            model_0(data)

            batch_time = time.time() - end
            end = time.time()

            len(train_loader) * epoch + batch_idx
            if batch_idx % args.log_interval == 0:
                num_samples = batch_idx * len(data)
                num_epochs = len(train_loader.dataset)
                percent_complete = 100.0 * batch_idx / len(train_loader)
                print(
                    f"Train Epoch: {epoch} [{num_samples}/{num_epochs} ({percent_complete:.0f}%)]\t"
                    f"Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                )

    avg_loss = avg_loss / len(train_loader)
    return avg_loss


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

    model.apply(lambda m: setattr(m, "t", args.t))
    model_0.apply(lambda m: setattr(m, "t", args.baset))
    model.apply(lambda m: setattr(m, "t1", args.t))
    model_0.apply(lambda m: setattr(m, "t1", args.baset))

    if args.update_bn:
        utils.update_bn(data_loader.train_loader, model, device=args.device)
        utils.update_bn(data_loader.train_loader, model_0, device=args.device)

    M = 20
    acc_bm_m0 = torch.zeros(M)
    conf_bm_m0 = torch.zeros(M)
    count_bm_m0 = torch.zeros(M)

    acc_bm_ens = torch.zeros(M)
    conf_bm_ens = torch.zeros(M)
    count_bm_ens = torch.zeros(M)

    acc_bm = torch.zeros(M)
    conf_bm = torch.zeros(M)
    count_bm = torch.zeros(M)

    with torch.no_grad():

        for data, target in val_loader:
            data, target = data.to(args.device), target.to(args.device)

            model.apply(lambda m: setattr(m, "t", args.t))
            model.apply(lambda m: setattr(m, "t1", args.t))
            output, feats = model(data)
            test_loss += criterion(output, target).item()

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct_vec = pred.eq(target.view_as(pred))
            correct += correct_vec.sum().item()

            # get model 0
            model_0.apply(lambda m: setattr(m, "t", args.baset))
            model_0.apply(lambda m: setattr(m, "t1", args.baset))
            model_0_output, model_0_feats = model_0(data)
            ensemble_output = (model_0_output + output) / 2
            ensemble_pred = ensemble_output.argmax(dim=1, keepdim=True)
            ensemble_correct_vec = ensemble_pred.eq(target.view_as(pred))
            ensemble_correct += ensemble_correct_vec.sum().item()

            m0_pred = model_0_output.argmax(dim=1, keepdim=True)
            m0_correct_vec = m0_pred.eq(target.view_as(pred))
            m0_correct += m0_correct_vec.sum().item()

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

            soft_out = output.softmax(dim=1)
            soft_out_m0 = model_0_output.softmax(dim=1)
            soft_out_ens = ensemble_output.softmax(dim=1)

            # need to do ece for m0, ensemble, model
            for i in range(data.size(0)):

                conf = soft_out[i][pred[i]]
                bin_idx = min((conf * M).int().item(), M - 1)
                acc_bm[bin_idx] += correct_vec[i].float().item()
                conf_bm[bin_idx] += conf.item()
                count_bm[bin_idx] += 1.0

                conf = soft_out_ens[i][pred[i]]
                bin_idx = min((conf * M).int().item(), M - 1)
                acc_bm_ens[bin_idx] += ensemble_correct_vec[i].float().item()
                conf_bm_ens[bin_idx] += conf.item()
                count_bm_ens[bin_idx] += 1.0

                conf = soft_out_m0[i][pred[i]]
                bin_idx = min((conf * M).int().item(), M - 1)
                acc_bm_m0[bin_idx] += m0_correct_vec[i].float().item()
                conf_bm_m0[bin_idx] += conf.item()
                count_bm_m0[bin_idx] += 1.0

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

    ece = 0.0
    for i in range(M):
        ece += (acc_bm[i] - conf_bm[i]).abs().item()
    ece /= len(val_loader.dataset)
    print("ece is", ece)

    ece_ens = 0.0
    for i in range(M):
        ece_ens += (acc_bm_ens[i] - conf_bm_ens[i]).abs().item()
    ece_ens /= len(val_loader.dataset)
    print("ece_ens is", ece_ens)

    ece_m0 = 0.0
    for i in range(M):
        ece_m0 += (acc_bm_m0[i] - conf_bm_m0[i]).abs().item()
    ece_m0 /= len(val_loader.dataset)
    print("ece_m0 is", ece_m0)

    metrics = {
        "ece": ece,
        "ece_ens": ece_ens,
        "ece_m0": ece_m0,
        "tvdist": tv_dist,
        "ensemble_acc": ensemble_acc,
        "feat_cossim": feat_cosim,
        "m0_acc": m0_acc,
    }

    return test_acc, metrics
