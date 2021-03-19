#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os
import pathlib
import time

import torch
import torch.backends.cudnn as cudnn

import models
from args import args


def set_gpu(model):
    if args.multigpu is None:
        args.device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )
        args.device = torch.cuda.current_device()
        cudnn.benchmark = True

    return model


def get_model():
    model = models.__dict__[args.model]()
    return model


def write_result_to_csv(**kwargs):
    results = pathlib.Path(args.log_dir) / "results.csv"
    tmp = "Date Finished,Name,curr_acc1,best_acc1,train_loss,".lower()
    if not results.exists():

        add = ",".join([k for k, v in kwargs.items() if k not in tmp])
        results.write_text(
            f"Date Finished,Name,curr_acc1,best_acc1,train_loss,{add}\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        s = ", ".join([str(v) for k, v in kwargs.items() if k not in tmp])
        f.write(
            (
                "{now}, "
                "{name}, "
                "{curr_acc1:.04f}, "
                "{best_acc1:.04f}, "
                "{train_loss:.04f}, "
                "{s}\n"
            ).format(now=now, s=s, **kwargs)
        )


def save_cpt(epoch, it, models, optimizers, best_acc1, curr_acc1):
    if not os.path.exists(args.run_base_dir / f"epoch_{epoch}_iter_{it}.pt"):
        torch.save(
            {
                "epoch": epoch,
                "iter": it,
                "arch": args.model,
                "state_dicts": [m.state_dict() for m in models],
                "optimizers": [
                    o.state_dict() for o in optimizers if o is not None
                ],
                "best_acc1": best_acc1,
                "curr_acc1": curr_acc1,
            },
            args.run_base_dir / f"epoch_{epoch}_iter_{it}.pt",
        )


def sd_to_vector(state_dict):
    vec = []
    for k in state_dict:
        if "num_batches" in k:
            continue
        vec.append(state_dict[k].view(-1))
    return torch.cat(vec)


def custom_sd_to_vector(state_dict, include, exclude):
    vec = []
    for k in state_dict:
        valid = True
        for e in exclude:
            if e in k:
                valid = False
        if not valid:
            continue

        valid = False
        for i in include:
            if i in k:
                valid = True
        if not valid:
            continue

        vec.append(state_dict[k].view(-1))
    return torch.cat(vec)


def vector_to_sd(vec, state_dict):
    pointer = 0
    for k in state_dict:
        if "num_batches" in k:
            continue
        num_param = state_dict[k].numel()
        state_dict[k] = vec[pointer : pointer + num_param].view_as(
            state_dict[k]
        )
        pointer += num_param


# modified from https://github.com/pytorch/contrib/blob/master/torchcontrib/optim/swa.py#L274
def update_bn(loader, model, device=None):
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


def get_optimizer(args, model):

    if args.optimizer == "sgd":

        parameters = list(model.named_parameters())
        parameters_to_opimizer = [v for n, v in parameters if v.requires_grad]
        if len(parameters_to_opimizer) == 0:
            return None
        optimizer = torch.optim.SGD(
            parameters_to_opimizer,
            args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
            nesterov=False,
        )
    else:
        print("Cant find opt.")
        exit()

    return optimizer
