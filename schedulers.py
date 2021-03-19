#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
# modified from https://github.com/allenai/hidden-networks/blob/master/utils/schedulers.py
import numpy as np

__all__ = ["multistep_lr", "cosine_lr", "constant_lr", "get_policy"]


def get_policy(name):
    if name is None:
        return constant_lr

    out_dict = {
        "constant_lr": constant_lr,
        "cosine_lr": cosine_lr,
        "multistep_lr": multistep_lr,
        "linear": linear_lr,
        "swa_constant_lr": swa_constant_lr,
        "swa_cyc_lr": swa_cyc_lr,
        "cyc_lr": cyc_lr,
        "lth_lr": lth_lr,
        "snapshot_cyc_lr": snapshot_cyc_lr,
    }

    return out_dict[name]


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def constant_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            lr = args.lr

        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def cosine_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            e = epoch - args.warmup_length
            es = args.epochs - args.warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * args.lr

        assign_learning_rate(optimizer, lr)
        print(lr)
        return lr

    return _lr_adjuster


def linear_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        lr = (1.0 - (epoch / args.epochs)) * args.lr

        assign_learning_rate(optimizer, lr)
        print("linear", lr)
        return lr

    return _lr_adjuster


def exp_increase(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        lr = (1.0 + (epoch / args.epochs)) * args.lr

        assign_learning_rate(optimizer, lr)
        print("linear", lr)
        return lr

    return _lr_adjuster


def multistep_lr(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    def _lr_adjuster(epoch, iteration):
        lr = args.lr * (args.lr_gamma ** (epoch // args.lr_adjust))

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length


def swa_constant_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            t = float(epoch / args.epochs)
            lr_ratio = args.swa_lr / args.lr
            if t <= 0.5:
                factor = 1.0
            elif t <= 0.9:
                factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
            else:
                factor = lr_ratio

            lr = args.lr * factor

        assign_learning_rate(optimizer, lr)
        print(lr)
        return lr

    return _lr_adjuster


# modified from https://github.com/timgaripov/swa
def swa_cyc_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            b = args.epochs
            # TODO: this only works for 160 epochs as of now.
            a = int(args.swa_start * 160)

            ts = [b]
            if args.num_models >= 2:
                ts = [
                    x
                    for x in np.arange(
                        a, b + 1, (b - a) // (args.num_models - 1)
                    )
                ]
                if len(ts) > args.num_models:
                    ts = [
                        x
                        for x in np.arange(
                            a, b, (b - a) // (args.num_models - 1)
                        )
                    ]
            if ts[-1] != b:
                ts[-1] = b

            if epoch < ts[0]:
                lrbase = args.lr
                e = epoch - args.warmup_length
                es = ts[0] - args.warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * lrbase
            else:
                for i in range(len(ts)):
                    if epoch >= ts[i]:
                        lrbase = args.swa_lr
                        e = epoch - ts[i]
                        es = ts[i + 1] - ts[i]
                        lr = 0.5 * (1 + np.cos(np.pi * e / es)) * lrbase

        assign_learning_rate(optimizer, lr)
        print(lr)
        return lr

    return _lr_adjuster


def snapshot_cyc_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            b = args.epochs
            a = args.warmup_length

            ts = [b]
            if args.num_models >= 2:
                ts = [
                    x for x in np.arange(a, b + 1, (b - a) // args.num_models)
                ]
            if ts[-1] != b:
                ts[-1] = b

            if epoch < ts[0]:
                lrbase = args.lr
                e = epoch - args.warmup_length
                es = ts[0] - args.warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * lrbase
            else:
                for i in range(len(ts)):
                    if epoch >= ts[i]:
                        lrbase = args.lr
                        e = epoch - ts[i]
                        es = ts[i + 1] - ts[i]
                        lr = 0.5 * (1 + np.cos(np.pi * e / es)) * lrbase

        assign_learning_rate(optimizer, lr)
        print(lr)
        return lr

    return _lr_adjuster


def cyc_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            b = args.epochs
            a = int(args.swa_start * 160)

            ts = [b]
            nm = 3
            if nm >= 2:
                ts = [x for x in np.arange(a, b + 1, (b - a) // (nm - 1))]
            if ts[-1] != b:
                ts[-1] = b

            if epoch < ts[0]:
                lrbase = args.lr
                e = epoch - args.warmup_length
                es = ts[0] - args.warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * lrbase
            else:
                for i in range(len(ts)):
                    if epoch >= ts[i]:
                        lrbase = args.swa_lr
                        e = epoch - ts[i]
                        es = ts[i + 1] - ts[i]
                        lr = 0.5 * (1 + np.cos(np.pi * e / es)) * lrbase

        assign_learning_rate(optimizer, lr)
        print(lr)
        return lr

    return _lr_adjuster


def lth_lr(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    def _lr_adjuster(epoch, iteration):
        if epoch < 80:
            lr = args.lr
        elif epoch < 120:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster
