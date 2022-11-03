"""Module for visualisations."""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCDIR = os.path.join(ROOTDIR, "docs")


def savefig(fig, name):
    """Save a matplotlib or seaborn figure in svg format.

    The function also calls tight_layout prior to saving.

    Args:
        fig: matplotlib.fig.Figure or seaborn.*Grid object
        name: str, name of figure without extension

    Returns:
        None

    """
    name = name + ".svg"
    fig.tight_layout()
    fig.savefig(os.path.join(DOCDIR, name), format="svg")


def gcfa(ncols=1, nrows=1, square=True, grid=None):

    if grid:
        g = sns.FacetGrid(
            data=grid.pop("data"),
            col=grid.pop("col"),
            col_wrap=grid.pop("col_wrap", 5),
            sharex=True,
            sharey=True,
        )
        fig, axs = g.fig, g.axes
    else:
        SCALER = 5 if square else 10
        W = SCALER * ncols
        H = 5
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(W, H),
            sharey=True,
        )

    return fig, axs


def heatmap(data, ax):
    corr = data.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(
        data=corr,
        mask=mask,
        square=True,
        annot=True,
        fmt=".3f",
        ax=ax,
    )
