#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import time

import torch

import utils
from args import args


def init(model, writer, train_loader):
    pass


def train(models, writer, data_loader, optimizers, criterion, epoch):

    # the default is to just train a single model
    model = models[0]
    optimizer = optimizers[0]

    model.zero_grad()
    model.train()
    avg_loss = 0.0
    train_loader = data_loader.train_loader

    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        data_time = time.time() - end

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()

        avg_loss += loss.item()

        batch_time = time.time() - end
        end = time.time()

        it = len(train_loader) * epoch + batch_idx
        if batch_idx % args.log_interval == 0:
            num_samples = batch_idx * len(data)
            num_epochs = len(train_loader.dataset)
            percent_complete = 100.0 * batch_idx / len(train_loader)
            print(
                f"Train Epoch: {epoch} [{num_samples}/{num_epochs} ({percent_complete:.0f}%)]\t"
                f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
            )

            if args.save:
                writer.add_scalar(f"train/loss", loss.item(), it)
        if args.save and it in args.save_iters:
            utils.save_cpt(epoch, it, models, optimizers, -1, -1)

    avg_loss = avg_loss / len(train_loader)
    return avg_loss


@torch.no_grad()
def test(models, writer, criterion, data_loader, epoch):

    model = models[0]

    model.zero_grad()
    model.eval()
    test_loss = 0
    correct = 0
    test_loader = data_loader.val_loader

    M = 20
    acc_bm = torch.zeros(M)
    conf_bm = torch.zeros(M)
    count_bm = torch.zeros(M)

    for data, target in test_loader:
        data, target = data.to(args.device), target.to(args.device)
        output = model(data)
        soft_out = output.softmax(dim=1)
        test_loss += criterion(output, target).item()
        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True)
        correct_vec = pred.eq(target.view_as(pred))
        correct += correct_vec.sum().item()

        for i in range(data.size(0)):
            conf = soft_out[i][pred[i]]
            bin_idx = min((conf * M).int().item(), M - 1)
            acc_bm[bin_idx] += correct_vec[i].float().item()
            conf_bm[bin_idx] += conf.item()
            count_bm[bin_idx] += 1.0

    test_acc = correct / len(test_loader.dataset)
    print(test_acc)

    ece = 0.0
    for i in range(M):
        ece += (acc_bm[i] - conf_bm[i]).abs().item()
    ece /= len(test_loader.dataset)

    return test_acc, {"ece": ece}
