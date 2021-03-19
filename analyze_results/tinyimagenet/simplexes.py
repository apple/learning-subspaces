#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import matplotlib.pyplot as plt

import viz.utils as utils
from viz.config import fs_helper
from viz.config import helper


def lvh_helper(ax, setxax=True, setyax=True):

    simplexes_data = utils.read_csv_files(
        [f"learning-subspaces-results/tinyimagenet/simplexes/results.csv",],
        ["curr_acc1"],
    )

    standard_data = utils.read_csv_files(
        [
            f"learning-subspaces-results/tinyimagenet/train-ensemble-members/results.csv"
        ],
        ["curr_acc1"],
    )

    xarr = [2, 3, 4]
    dh = utils.DataHolder(xarr, r"\alpha", "one-dim-subspaces")

    utils.add_data_helper(
        dh, ax, simplexes_data, "n", "curr_acc1", id="simplex", **helper("m_6")
    )
    utils.add_data_helper(
        dh,
        ax,
        simplexes_data,
        "n",
        "curr_acc1",
        id="simplex-layerwise",
        **helper("m_4"),
    )

    baselines = utils.query(standard_data, x="ln", y="curr_acc1",)

    dh.add(
        *utils.add_baseline(
            ax, xarr, baselines[0.0], **helper("standard_training")
        )
    )

    if setxax:
        ax.set_xlabel(r"$\alpha$", fontsize=fs_helper("xlabel"))
    if setyax:
        ax.set_ylabel("Accuracy", fontsize=fs_helper("ylabel"))

    # ax.set_ylim([0.625,0.675])


if __name__ == "__main__":

    save = None

    utils.format(plt)

    fig, axlist = plt.subplots(1, 1, figsize=(6, 6))
    ax = axlist

    lvh_helper(ax)

    fig.subplots_adjust(
        top=0.97, left=0.07, right=0.9, bottom=0.3, wspace=0.15, hspace=0.23
    )

    legend = ax.legend(
        loc="upper center",
        # bbox_to_anchor = (0.5, -0.2),
        bbox_to_anchor=(0.425, -0.25),
        ncol=5,
        fontsize=fs_helper("legend"),
    )

    # plt.show()
    plt.savefig("simplexes.pdf", bbox_inches="tight")
