#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
info = {
    "ens_6": ["C0", "-", "x", "Line (Ensemble)"],
    "ens_6_28": ["C0", "-", "x", r"Line (Ensemble $\alpha=0.2,0.8$)"],
    "ens_4": ["C1", "-", "d", "Line (Layerwise, Ensemble)"],
    "ens_0": ["C3", "-", "o", "Curve (Ensemble)"],
    "ens_1": ["C4", "-", "^", "Curve (Ensemble)"],
    "t_6": ["C0", "--", "x", "Line"],
    "t_4": ["C1", "--", "d", "Line (Layerwise)"],
    "t_0": ["C3", "--", "o", "Curve"],
    "t_1": ["C4", "--", "^", "Curve"],
    "l_6": ["C0", "--", "x", "Line (Midpoint)"],
    "m_6": ["C0", "--", "x", "Simplex (Midpoint)"],
    "m_4": ["C1", "--", "d", "Simplex (Layerwise, Midpoint)"],
    "swa_swa_constant_lr": ["C2", "--", "o", "SWA (High Const. LR)"],
    "swa_swa_cyc_lr": ["k", "--", "s", "SWA (Cyclic LR)"],
    "standard_training": ["k", "--", None, "Standard Training"],
    "early_stop": ["k", ":", None, "Standard Training (Opt. Early Stop)"],
    "dropout": ["C5", "-.", None, "Dropout (best)"],
    "smooth": ["C4", "-.", None, r"Label Smoothing (best)"],
    "standard_ensemble": ["k", ":", None, "Standard Ensemble of Two"],
    "simplex": ["C6", "--", "x", "Simplex (Ensemble)"],
    "ens_6_300": ["C0", "-", "x", "Line (Ensemble, 300 Epochs)"],
    "t_6_300": ["C0", "--", "x", "Line (300 Epochs)"],
}


def lookup(c):
    if c == "k":
        return "#A8A8A8"
    else:
        return c


def helper(name):
    return {
        "color": lookup(info[name][0]),
        "ls": info[name][1],
        "marker": info[name][2],
        "label": info[name][3],
    }


fsinfo = {
    "legend": 18,
    "xlabel": 19,
    "ylabel": 19,
    "title": 18,
}


def fs_helper(name):
    return fsinfo[name]
