#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import argparse

args = None


def parse_arguments():
    # Training settings
    parser = argparse.ArgumentParser(description="Learning-Subspaces")
    parser.add_argument(
        "--num-models",
        type=int,
        default=1,
        help="Number of models currently being considered for training ensembles or SWA",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples drawn from the subspace for each batch.",
    )
    parser.add_argument(
        "--optimizer", type=str, default="sgd", help="Which optimizer to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for testing (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=160,
        metavar="N",
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="Momentum (default: 0.9)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5,
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0001,
        metavar="M",
        help="Weight decay (default: 0.0001)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="how many cpu workers"
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=10,
        help="how many total neurons in last layer",
    )
    parser.add_argument(
        "--name", type=str, default="default", help="Experiment id."
    )
    parser.add_argument(
        "--data", type=str, help="Location to store data",
    )
    parser.add_argument(
        "--log-dir", type=str, help="Location to logs/checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=lambda x: [str(a) for a in x.split(",")],
        default=None,
        help="optionally resume",
    )
    parser.add_argument(
        "--model-name", type=str, default=None, help="model name where required"
    )
    parser.add_argument(
        "--ct", type=str, default="snow", help="Corruption type for ImageNet-C"
    )
    parser.add_argument(
        "--sev", type=int, default=1, help="Corruption severity for ImageNet-C"
    )

    parser.add_argument(
        "--width-mult", type=float, default=1.0, help="how wide is each layer"
    )
    parser.add_argument(
        "--conv_type",
        type=str,
        default="StandardConv",
        help="Type of conv layer",
    )
    parser.add_argument(
        "--bn_type",
        type=str,
        default="StandardBN",
        help="Type of batch norm layer.",
    )

    parser.add_argument(
        "--conv-init",
        type=str,
        default="kaiming_normal",
        help="How to initialize the conv weights.",
    )

    parser.add_argument(
        "--baset", type=float, default=0.1,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=-1,
        help="For simplexes -- number of endpoints used to define the simplex.",
    )
    parser.add_argument("--model", type=str, help="Type of model.")
    parser.add_argument(
        "--multigpu",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="Which GPUs to use for multigpu training",
    )
    parser.add_argument(
        "--save-epochs",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="Which epochs to save",
    )
    parser.add_argument(
        "--save-iters",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="Which epochs to save",
    )
    parser.add_argument(
        "--swa-save-epochs",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="Which epochs to save for swa",
    )
    parser.add_argument(
        "--mode", default="fan_in", help="Weight initialization mode"
    )
    parser.add_argument(
        "--nonlinearity",
        default="relu",
        help="Nonlinearity used by initialization",
    )
    parser.add_argument("--set", type=str, help="Which dataset to use")
    parser.add_argument(
        "--trainer",
        default=None,
        type=str,
        help="Which trainer to use, default in trainers/default.py",
    )
    parser.add_argument("--lr-policy", default=None, help="Scheduler to use")
    parser.add_argument(
        "--warmup-length", default=0, type=int,
    )
    parser.add_argument(
        "--pretrained", action="store_true", default=False,
    )
    parser.add_argument(
        "--save", action="store_true", default=False,
    )
    parser.add_argument(
        "--save-data", action="store_true", default=False,
    )
    parser.add_argument(
        "--trainswa", action="store_true", default=False,
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=None,
    )
    parser.add_argument(
        "--label-noise", type=float, default=None,
    )
    parser.add_argument(
        "--beta", type=float, default=-1,
    )
    parser.add_argument(
        "--lamb", type=float, default=-1,
    )
    parser.add_argument(
        "--swa-start", type=float, default=0.75,
    )
    parser.add_argument(
        "--swa-lr", type=float, default=0.05,
    )
    parser.add_argument(
        "--test-freq", type=int, default=None,
    )
    parser.add_argument(
        "--update-bn", action="store_true", default=False,
    )
    parser.add_argument(
        "--train-update-bn", action="store_true", default=False,
    )

    args = parser.parse_args()
    return args


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
