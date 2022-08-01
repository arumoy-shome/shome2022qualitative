"""General purpose variables & functions.

This module provides general purpose functions & variables that can be
used across other modules and analysis documents.

"""


def savefig(fig, name):
    """Save a matplotlib figure in pdf & png format. This function
    also calls the tight_layout function prior to saving.

    In:
    ---
    fig: matplotlib.fig.Figure object
    name: str, name of figure without extension

    Out:
    ----
    name: str, name of the figure with png extension.

    """
    fig.tight_layout()
    fig.savefig(name + ".png", format="png")
    fig.savefig(name + ".pdf", format="pdf")

    return name + ".png"
