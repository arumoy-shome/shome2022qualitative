"""Module for generating visualisations.

This module provides functions to generate visualisations for the data
generated by bin/data.py.

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATADIR = os.path.join(ROOTDIR, "data")
DOCDIR = os.path.join(ROOTDIR, "docs")

PRIVILEGED = ["None", "True", "False"]
MODELS = [("logisticregression", "lr"), ("decisiontreeclassifier", "dt")]


def savefig(fig, name):
    """Save a matplotlib figure in pdf & png format. This function
    also calls the tight_layout function prior to saving.

    Args:
        fig: matplotlib.fig.Figure object
        name: str, name of figure without extension

    Returns:
        None

    """
    fig.tight_layout()
    # fig.savefig(name + ".png", format="png")
    # fig.savefig(name + ".pdf", format="pdf")
    fig.savefig(name + ".svg", format="svg")


def precision_recall(dataset, protected, data):
    """Visualise precision & recall.

    Args:
        dataset: Str, name of dataset
        protected: Str, name of protected attribute
        data: pandas.Dataframe

    Returns:
        None
    """
    cols = ["accuracy", "PPV", "TPR", "f1"]

    for attr in protected:
        df = data[data["protected"] == attr]

        for model, abrev in MODELS:
            name = "{}_barplot_prot-{}_mod-{}_acc-pre-rec-f1".format(
                dataset, attr, abrev
            )
            fig, axs = plt.subplots(1, len(cols), sharey=True, figsize=(20, 5))
            metrics = df[df["model"] == model]

            for idx, col in enumerate(cols):
                sns.barplot(
                    data=metrics,
                    y=col,
                    x="subset",
                    hue="privileged",
                    hue_order=PRIVILEGED,
                    ax=axs[idx],
                )
            axs[cols.index("PPV")].set_ylabel("precision")
            axs[cols.index("TPR")].set_ylabel("recall")

            for idx in range(len(cols)):
                for container in axs[idx].containers:
                    axs[idx].bar_label(container)

            savefig(fig, os.path.join(DOCDIR, name))


def confusion_matrix(dataset, protected, data, rate):
    """Visualise confusion matrix.

    Args:
        dataset: Str, name of dataset
        protected: Str, name of protected attribute
        data: pandas.Dataframe
        rate: Bool, visualise absolute numbers or rate

    Returns:
        None
    """
    for attr in protected:
        df = data[data["protected"] == attr]
        if rate:
            r = "-rate"
            fmt = ".3f"
            cols = ["TNR", "FPR", "FNR", "TPR"]
        else:
            r = ""
            fmt = ""
            cols = ["TN", "FP", "FN", "TP"]

        for model, abrev in MODELS:
            name = "{}_heatmap_prot-{}_mod-{}_cm{}".format(dataset, attr, abrev, r)
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            metrics = df[df["model"] == model]

            for idx, privileged in enumerate(PRIVILEGED):
                cm = metrics[metrics["privileged"] == privileged]
                cm = cm[cols].values.reshape(2, 2)

                sns.heatmap(
                    data=cm, annot=cm, fmt=fmt, cbar=False, cmap="Blues", ax=axs[idx]
                )
                axs[idx].set_xlabel("y_pred")
                axs[idx].set_ylabel("y_true")
                axs[idx].set_title(privileged)

            savefig(fig, os.path.join(DOCDIR, name))


def disparate_impact(dataset, protected, data, model):
    """Visualise disparate impact.

    Args:
        dataset: Str, name of dataset
        protected: Str, name of protected attribute
        data: pandas.Dataframe
        model: Bool, visualise metric for model vs. data

    Returns:
        None
    """
    for attr in protected:
        # data disparate impact
        df = data[data["protected"] == attr]
        mod = "all" if model else "none"
        name = "{}_barplot_prot-{}_mod-{}_disparate-impact".format(dataset, attr, mod)
        fig, ax = plt.subplots()

        sns.barplot(
            data=df,
            y="disparate_impact",
            x="model" if model else "subset",
            ax=ax,
            ci=None,
        )

        for container in ax.containers:
            ax.bar_label(container)

        savefig(fig, os.path.join(DOCDIR, name))


def base_rate(dataset, protected, data):
    """Visualise base rate.

    Args:
        dataset: Str, name of dataset
        protected: Str, name of protected attribute
        data: pandas.Dataframe

    Returns:
        None
    """
    for attr in protected:
        df = data[data["protected"] == attr]
        name = "{}_barplot_prot-{}_base-rate".format(dataset, attr)
        fig, ax = plt.subplots()

        sns.barplot(
            data=df,
            y="base_rate",
            x="subset",
            hue="privileged",
            hue_order=PRIVILEGED,
            ax=ax,
            ci=None,
        )

        for container in ax.containers:
            ax.bar_label(container)

        savefig(fig, os.path.join(DOCDIR, name))


def num_pos_neg(dataset, protected, data):
    """Visualise num_positives & num_negatives.

    Args:
        dataset: Str, name of dataset
        protected: Str, name of protected attribute
        data: pandas.Dataframe

    Returns:
        None
    """
    for attr in protected:
        df = data[data["protected"] == attr]
        name = "{}_barplot_prot-{}_subset-all_num-pos-neg".format(dataset, attr)
        cols = ["num_positives", "num_negatives"]
        fig, axs = plt.subplots(1, len(cols), sharey=True, figsize=(10, 5))

        for idx, col in enumerate(cols):
            sns.barplot(
                data=df,
                y=col,
                x="subset",
                hue="privileged",
                hue_order=PRIVILEGED,
                ax=axs[idx],
                ci=None,
            )

        for idx in range(len(cols)):
            for container in axs[idx].containers:
                axs[idx].bar_label(container)

        savefig(fig, os.path.join(DOCDIR, name))


if __name__ == "__main__":
    datasets = [
        ("adult", ["sex", "race"]),
        ("compas", ["sex", "race"]),
        ("bank", ["age"]),
        ("german", ["sex", "age"]),
        ("meps", ["race"]),
    ]
    df = pd.read_csv(os.path.join(DATADIR, "data.csv"))

    for dataset, protected in datasets:
        data = df[df["dataset"] == dataset]

        # data fairness metrics
        num_pos_neg(dataset=dataset, protected=protected, data=data)
        base_rate(dataset=dataset, protected=protected, data=data)
        disparate_impact(
            dataset=dataset,
            protected=protected,
            data=data[data["model"] == "None"],
            model=False,
        )

        # performance metrics
        confusion_matrix(dataset=dataset, protected=protected, data=data, rate=False)
        confusion_matrix(dataset=dataset, protected=protected, data=data, rate=True)
        precision_recall(dataset=dataset, protected=protected, data=data)

        # model fairness metrics
        disparate_impact(
            dataset=dataset,
            protected=protected,
            data=data[data["subset"] == "test"],
            model=True,
        )
