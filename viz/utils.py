#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import csv
import pathlib

import numpy as np
import pandas as pd


def try_float(x):
    try:
        return int(x)
    except:
        try:
            return float(x)
        except:
            return x


# keys is a list of strings k_1,...,k_n
# returns a dict d where d[name][k_i] = val corresponding to k_i
def read_csv_files(files, keys):
    d = {}
    for f in files:
        with open(f, mode="r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                name = row["Name"] if "Name" in row else row[" Name"]
                d[name] = {}
                for k in keys:
                    d[name][k] = try_float(row[k]) if k != "wd" else row[k]
    return d


def id_to_dict(id):
    if "+" in id:
        l = [x.split("=") for x in id.split("+")]
    else:
        l = [x.split("=") for x in id.split("~")]
    l = [x for x in l if len(x) == 2]
    return {x[0].strip(): try_float(x[1]) if x[0] != "wd" else x[1] for x in l}


def dict_to_id(dict):
    return "~".join("{}={}".format(k, v) for k, v in dict.items())


def query(data, x, y, outlier=lambda x, y, d: False, **kwargs):
    out = {}
    for id in data:
        dict = id_to_dict(id)
        valid = True
        for k, v in kwargs.items():
            if v is None:
                if k in dict:
                    valid = False
                    break
            else:
                if k not in dict or dict[k] != v:
                    valid = False
                    break
        if valid:
            if x in dict:
                x_ = dict[x]
                y_ = data[id][y]
                if not outlier(x_, y_, data):
                    if x_ not in out:
                        out[x_] = []
                    out[x_].append(y_)
                else:
                    print(y_)

    return out


def add_curve(ax, data, label, ls="-", marker=None, color=None):
    xs = np.array(sorted([k for k in data]))
    ys = np.array([np.mean(data[x]) for x in xs])
    std = np.array([np.std(data[x]) for x in xs])
    ax.plot(
        xs,
        ys,
        label=label,
        linestyle=ls,
        marker=marker,
        color=color,
        linewidth=3,
        alpha=0.9,
    )
    # ax.fill_between(xs, ys - std, ys + std, alpha=0.5, color=color)

    return xs, ys, std, label


def add_curve_with_noise(
    ax, data, noise, label, ls="-", marker=None, color=None
):
    xs = np.array(sorted([k for k in data]))
    ys = np.array([np.mean(data[x]) for x in xs])
    noise_ys = np.array([np.mean(noise[x]) for x in xs])
    ax.plot(xs, ys, label=label, linestyle=ls, marker=marker, color=color)
    ax.fill_between(xs, ys - noise_ys, ys + noise_ys, alpha=0.5, color=color)

    return xs, ys, noise, label


def add_curves(
    ax,
    data,
    x,
    y,
    key,
    vals,
    outlier=lambda x, y, d: False,
    colors=None,
    ls=None,
    prefix="",
    **kwargs,
):

    if colors is None:
        colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    if ls is None:
        ls = ["-"]
    i = 0
    for val in vals:
        kwargs[key] = val
        q = query(data, x=x, y=y, outlier=outlier, **kwargs)
        add_curve(
            ax,
            q,
            "{}{}={}".format(prefix, key, val),
            ls=ls[i % len(ls)],
            color=colors[i % len(colors)],
        )
        i += 1


def clean(data):
    try:
        return [d for d in data if not np.isnan(d)]
    except Exception:
        return


def add_point(ax, data, label, ls="-", marker=None, color=None, linewidth=1):
    xs = np.array(sorted([k for k in data]))
    ys = np.array([np.mean(clean(data[x])) for x in xs])
    std = np.array([np.std(clean(data[x])) for x in xs])
    # ax.plot(xs, ys, label=label, linestyle=ls, marker=marker, color=color)
    if label is None:
        ax.errorbar(
            xs,
            ys,
            yerr=(std, std),
            marker=marker,
            markersize=7,
            linewidth=linewidth,
            linestyle=ls,
            capsize=1,
            dashes=(5, 5),
            capthick=0.2,
            color=color,
            alpha=0.75,
        )

    else:
        # if label == 'Line' or label == 'Curve' or label == 'Line (Layerwise)' or label == 'SWA (Cyclic LR)':
        #     ax.errorbar(xs, ys, yerr=(std, std), label=label, marker=marker, markersize=7, dashes = (3,4), linewidth=linewidth, linestyle=ls, capsize=1, capthick=0.2, color=color, alpha=0.75)

        # elif 'Line' in label and 'Ensemble' not in label:
        #     ax.errorbar(xs, ys, yerr=(std, std), label=label, marker=marker, markersize=7, dashes = (4,4), linewidth=linewidth, linestyle=ls, capsize=1, capthick=0.2, color=color, alpha=0.75)

        # else:
        ax.errorbar(
            xs,
            ys,
            yerr=(std, std),
            label=label,
            marker=marker,
            markersize=7,
            linewidth=linewidth,
            linestyle=ls,
            capsize=1,
            capthick=0.2,
            color=color,
            alpha=0.75,
        )

    return xs, ys, std, label


def add_point_with_noise(
    ax, data, std_data, label, ls="-", marker=None, color=None, linewidth=1
):
    xs = np.array(sorted([k for k in data]))
    ys = np.array([np.mean(data[x]) for x in xs])
    std = np.array([np.mean(std_data[x]) for x in xs])
    # ax.plot(xs, ys, label=label, linestyle=ls, marker=marker, color=color)
    if label is None:
        ax.errorbar(
            xs,
            ys,
            yerr=(std, std),
            marker=marker,
            markersize=7,
            linewidth=linewidth,
            linestyle=ls,
            capsize=4,
            capthick=0.5,
            color=color,
        )

    else:
        ax.errorbar(
            xs,
            ys,
            yerr=(std, std),
            label=label,
            marker=marker,
            markersize=7,
            linewidth=linewidth,
            linestyle=ls,
            capsize=4,
            capthick=0.5,
            color=color,
        )

    return xs, ys, std, label


def add_pointv2(ax, data, label, ls="-", marker=None, color=None, linewidth=1):
    xs = np.array(sorted([k for k in data]))
    ys = np.array([np.mean(data[x]) for x in xs])

    std_p = np.array(
        [
            min(np.std(data[x]), np.max(np.array(data[x])) - np.mean(data[x]))
            for x in xs
        ]
    )
    std_n = np.array(
        [
            min(np.std(data[x]), np.mean(data[x]) - np.min(np.array(data[x])))
            for x in xs
        ]
    )
    # ax.plot(xs, ys, label=label, linestyle=ls, marker=marker, color=color)
    ax.errorbar(
        xs,
        ys,
        yerr=(std_n, std_p),
        label=label,
        marker=marker,
        linewidth=linewidth,
        linestyle=ls,
        capsize=5,
        capthick=1,
        color=color,
    )

    return xs, ys, np.minimum(std_n, std_p), label


def add_upper_bound(
    ax, data, label, range, ls="-", marker=None, color=None, linewidth=1
):
    x0, xT = range
    xs = np.array(sorted([k for k in data]))
    ys = np.array([np.mean(data[x]) for x in xs])
    np.array([np.std(data[x]) for x in xs])

    overall_mean = np.mean(ys)
    new_xs = np.arange(x0, xT)
    ys = [overall_mean for _ in new_xs]
    ax.plot(
        new_xs,
        ys,
        label=label,
        marker=marker,
        linestyle=ls,
        color=color,
        linewidth=linewidth,
    )

    return new_xs, ys, [0 for _ in new_xs], label


def add_baseline(
    ax, r, rd, label=None, color=None, marker=None, ls=None, **kwargs
):
    mu = np.mean(rd)
    sig = np.std(rd)
    xs, ys, std = [i for i in r], [mu for _ in r], [sig for _ in r]

    ax.plot(
        [i for i in r],
        [mu for _ in r],
        linestyle=ls,
        color=color,
        label=label,
        marker=marker,
        linewidth=3,
    )
    ax.fill_between(
        [i for i in r],
        [mu - sig for _ in r],
        [mu + sig for _ in r],
        color=color,
        alpha=0.3,
    )

    return xs, ys, std, label


def format(plt, nomp=False):
    SMALL_SIZE = 16
    MEDIUM_SIZE = 24

    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    # plt.style.use('seaborn-paper')
    # plt.style.use('dark_background')
    plt.rc("text.latex", preamble=r"\usepackage{bbold}")
    if nomp:
        return
    from matplotlib import rc

    rc("text", usetex=True)


def add_data_helper(dh, ax, data, x, y, label, **kwargs):
    if "ls" in kwargs:
        ls = kwargs["ls"]
        del kwargs["ls"]
    if "marker" in kwargs:
        marker = kwargs["marker"]
        del kwargs["marker"]
    if "color" in kwargs:
        color = kwargs["color"]
        del kwargs["color"]
    if "outlier" in kwargs:
        outlier = kwargs["outlier"]
        del kwargs["outlier"]
    else:
        outlier = lambda x, y, d: False
    if "reverse" in kwargs:
        reverse = kwargs["reverse"]
        del kwargs["reverse"]
    else:
        reverse = False
    if "mirror" in kwargs:
        mirror = kwargs["mirror"]
        del kwargs["mirror"]
    else:
        mirror = False
    if "preprend" in kwargs:
        preprend = kwargs["preprend"]
        del kwargs["preprend"]
    else:
        preprend = None
    curve = query(data, x=x, y=y, outlier=outlier, **kwargs,)
    if preprend is not None:
        curve[1] = preprend[1]
    if reverse:
        if "acc" in y:
            new_curve = query(data, x=x, y="m0_acc", outlier=outlier, **kwargs)
        else:
            new_curve = query(data, x=x, y="ece_m0", outlier=outlier, **kwargs)
        curve.update({1 - k: v for k, v in new_curve.items()})
    if mirror:
        curve.update({1 - k: v for k, v in curve.items()})

    dh.add(
        *add_point(
            ax,
            curve,
            label=label,
            color=color,
            ls=ls,
            linewidth=3,
            marker=marker,
        )
    )

    return curve


class DataHolder:
    def __init__(self, xarr, xtitle, plottitle, include_std=False):
        self.xarr = xarr
        self.xtitle = xtitle
        self.ys = {}
        self.stds = {}
        self.mean = {}
        self.plottitle = plottitle
        self.include_std = include_std
        if self.include_std:
            self.std = {}

    def add(self, xs, ys, std, label):
        valid = False
        y_data = {}
        std_data = {}
        xs = [round(x, 2) for x in xs]
        for i, x in enumerate(xs):
            if x in self.xarr:
                valid = True
                y_data[x] = ys[i]
                std_data[x] = std[i]

        if valid:
            self.mean[label] = np.mean(np.array([v for k, v in y_data.items()]))
            if self.include_std:
                self.std[label] = np.std(
                    np.array([v for k, v in y_data.items()])
                )
            for x in self.xarr:
                if x not in y_data:
                    y_data[x] = "-"
                    std_data[x] = "-"

            self.ys[label] = [y_data[x] for x in self.xarr]
            self.stds[label] = [std_data[x] for x in self.xarr]

    def write(self, dir):
        results = pathlib.Path(dir) / f"{self.plottitle}.csv"
        header = "Entry"
        for x in self.xarr:
            header = header + f",{x}"
        if self.include_std:
            header = header + ",Avg,Std\n"
        else:
            header = header + ",Avg\n"
        results.write_text(header)
        for label in self.ys:
            entry = label.replace(",", "")
            for y in self.ys[label]:
                entry = entry + f",{y:.4f}" if y != "-" else entry + f",{y}"
            if self.include_std:
                entry = (
                    entry + f",{self.mean[label]:.4f},{self.std[label]:.4f}\n"
                )
            else:
                entry = entry + f",{self.mean[label]:.4f}\n"
            with open(results, "a+") as f:
                f.write(entry)

        # ok now also write a latex
        df = pd.read_csv(results)
        results = pathlib.Path(dir) / f"{self.plottitle}.tex"
        results.write_text(df.to_latex(index=False))
