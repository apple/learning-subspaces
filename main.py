#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import os
import pathlib
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import data
import schedulers
import trainers
import utils
from args import args


def main():
    # Seed.
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    # If saving models or saving data, create a folder for storing these files.
    # args.save => saving models, tensorboard logs, etc.
    # args.save_data => just saving results.
    if args.save or args.save_data:
        i = 0
        while True:
            run_base_dir = pathlib.Path(
                f"{args.log_dir}/{args.name}+try={str(i)}"
            )

            if not run_base_dir.exists():
                os.makedirs(run_base_dir)
                args.name = args.name + f"+try={i}"
                break
            i += 1

        (run_base_dir / "settings.txt").write_text(str(args))
        args.run_base_dir = run_base_dir

        print(f"=> Saving data in {run_base_dir}")

    # Get dataloader.
    data_loader = getattr(data, args.set)()

    curr_acc1 = 0.0

    # Make a list of models, instead of a single model.
    # This is not for training subspaces, but rather for the ensemble & SWA baselines.
    models = [utils.get_model() for _ in range(args.num_models)]

    # when training the SWA baseline, turn off the gradient to all but the first model.
    if args.trainswa:
        for i in range(1, args.num_models):
            for p in models[i].parameters():
                p.requires_grad = False

    # Resume a model from a saved checkpoint.
    num_models_filled = 0
    num_models = -1
    if args.resume:
        for i, resume in enumerate(args.resume):
            if type(resume) == tuple:
                # can use a tuple to provide how many models to load.
                resume, num_models = resume

            if os.path.isfile(resume):
                print(f"=> Loading checkpoint '{resume}'")
                checkpoint = torch.load(resume, map_location="cpu")

                pretrained_dicts = [
                    {k[7:]: v for k, v in c.items()}
                    for c in checkpoint["state_dicts"]
                ]
                n = 0
                for pretrained_dict in pretrained_dicts:
                    print(num_models_filled)
                    model_dict = models[num_models_filled].state_dict()
                    pretrained_dict = {
                        k: v
                        for k, v in pretrained_dict.items()
                        if k in model_dict
                    }
                    model_dict.update(pretrained_dict)
                    models[num_models_filled].load_state_dict(model_dict)
                    num_models_filled += 1
                    n += 1
                    if num_models > 0 and n >= num_models:
                        break

                print(
                    f"=> Loaded checkpoint '{resume}' (epoch {checkpoint['epoch']})"
                )
            else:
                print(f"=> No checkpoint found at '{resume}'")

    # Put models on the GPU.
    models = [utils.set_gpu(m) for m in models]

    # Get training loss.
    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss()
    else:
        print("adding label smoothing!")
        criterion = LabelSmoothing(smoothing=args.label_smoothing)
    criterion = criterion.to(args.device)

    if args.save:
        writer = SummaryWriter(log_dir=run_base_dir)
    else:
        writer = None

    # Get the "trainer", which specified how the model is trained.
    trainer = getattr(trainers, args.trainer or "default")
    print(f"=> Using trainer {trainer}")

    train, test = trainer.train, trainer.test

    # Call "init" on the trainer.
    trainer.init(models, writer, data_loader)

    # Since we have have a list of models, we also use a list of optimizers & schedulers.
    # When training subspaces, this list is of length 1.
    metrics = {}
    optimizers = [utils.get_optimizer(args, m) for m in models]
    lr_schedulers = [
        schedulers.get_policy(args.lr_policy or "cosine_lr")(o, args)
        for o in optimizers
        if o is not None
    ]

    # more logic for resuming a checkpoint, specifically concerned with the "pretrained" argument.
    # if args.pretrained, then we are not resuming. This means that we start from epoch 0.
    # if not args.pretrained, we are resuming and have to set the epoch, etc. appropriately.
    init_epoch = 0
    num_models_filled = 0
    if args.resume and not args.pretrained:
        for i, resume in enumerate(args.resume):
            if os.path.isfile(resume):
                print(f"=> Loading checkpoint '{resume}'")
                checkpoint = torch.load(resume, map_location="cpu")
                init_epoch = checkpoint["epoch"]
                curr_acc1 = checkpoint["curr_acc1"]
                for opt in checkpoint["optimizers"]:
                    if args.late_start >= 0:
                        continue
                    optimizers[num_models_filled].load_state_dict(opt)
                    num_models_filled += 1

    best_acc1 = 0.0
    train_loss = 0.0

    # Save the initialization.
    if init_epoch == 0 and args.save:
        print("saving checkpoint")
        utils.save_cpt(init_epoch, 0, models, optimizers, best_acc1, curr_acc1)

    # If the start epoch == the end epoch, just do evaluation "test".
    if init_epoch == args.epochs:
        curr_acc1, metrics = test(
            models, writer, criterion, data_loader, init_epoch,
        )

        if args.save or args.save_data:
            metrics["epoch"] = init_epoch
            utils.write_result_to_csv(
                name=args.name + f"+curr_epoch={init_epoch}",
                curr_acc1=curr_acc1,
                best_acc1=best_acc1,
                train_loss=train_loss,
                **metrics,
            )

    # Train from init_epoch -> args.epochs.
    for epoch in range(init_epoch, args.epochs):
        for lr_scheduler in lr_schedulers:
            lr_scheduler(epoch, None)
        train_loss = train(
            models, writer, data_loader, optimizers, criterion, epoch,
        )
        if type(train_loss) is tuple:
            train_loss, optimizers = train_loss

        if (
            args.test_freq is None
            or (epoch % args.test_freq == 0)
            or epoch == args.epochs - 1
        ):
            curr_acc1, metrics = test(
                models, writer, criterion, data_loader, epoch,
            )
        if curr_acc1 > best_acc1:
            best_acc1 = curr_acc1

        metrics["epoch"] = epoch + 1

        # This is for the SWA baseline -- we need to lookup if this an epoch for which we are saving a checkpoint.
        # If so we save a checkpoint and move it to the corresponding place in the models list.
        if args.trainswa and (epoch + 1) in args.swa_save_epochs:
            j = args.swa_save_epochs.index(epoch + 1)
            for m1, m2 in zip(models[0].modules(), models[j].modules()):
                if isinstance(m1, nn.Conv2d):
                    m2.weight = nn.Parameter(m1.weight.clone().detach())
                    m2.weight.requires_grad = False
                elif isinstance(m1, nn.BatchNorm2d):
                    m2.weight = nn.Parameter(m1.weight.clone().detach())
                    m2.bias = nn.Parameter(m1.bias.clone().detach())
                    m2.weight.requires_grad = False
                    m2.bias.requires_grad = False

        # Save checkpoint.
        if (
            args.save
            and args.save_epochs is not None
            and (epoch + 1) in args.save_epochs
        ):
            it = (epoch + 1) * len(data_loader.train_loader)
            utils.save_cpt(
                epoch + 1, it, models, optimizers, best_acc1, curr_acc1
            )

    # Save results.
    if args.save or args.save_data:
        utils.write_result_to_csv(
            name=args.name,
            curr_acc1=curr_acc1,
            best_acc1=best_acc1,
            train_loss=train_loss,
            **metrics,
        )

    # Save final checkpiont.
    if args.save:
        it = args.epochs * len(data_loader.train_loader)
        utils.save_cpt(
            args.epochs, it, models, optimizers, best_acc1, curr_acc1
        )

    return curr_acc1, metrics


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


if __name__ == "__main__":
    main()
