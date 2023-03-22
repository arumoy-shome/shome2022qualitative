"""Experiments with distribution change."""

from aif360.datasets import (
    AdultDataset,
    CompasDataset,
    BankDataset,
    GermanDataset,
    MEPSDataset21,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import os
import sys
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATADIR = os.path.join(ROOTDIR, "data")

sys.path.insert(0, ROOTDIR)
from src.utils import write_csv
from src.metrics import compute_data_metrics, compute_model_metrics

MODELS = [
    None,
    LogisticRegression,
    DecisionTreeClassifier,
    AdaBoostClassifier,
    RandomForestClassifier,
]
PRIVILEGED = [None, True, False]

PRIVILEGED_CLASSES_MAP = {
    "adult": {"sex": [["Male"]], "race": [["White"]]},
    "compas": {"sex": [["Female"]], "race": [["Caucasian"]]},
    "bank": {"age": [lambda x: x > 25]},
    "german": {"sex": [["male"]], "age": [lambda x: x > 25]},
    "meps": {"RACE": [["White"]]},
}

DATASET_MAP = {
    "adult": AdultDataset,
    "compas": CompasDataset,
    "bank": BankDataset,
    "german": GermanDataset,
    "meps": MEPSDataset21,
}

FEATURES = None
with open(os.path.join(ROOTDIR, "src", "features.json")) as f:
    FEATURES = json.load(f)


def parse_args():
    """Parse command line arguments.

    In:
        None

    Returns:
        args: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        help="Dataset & protected attribute in <dataset-pattr> format",
        choices=[
            "adult-sex",
            "adult-race",
            "compas-sex",
            "compas-race",
            "bank-age",
            "german-sex",
            "german-age",
            "meps-RACE",
        ],
    )
    parser.add_argument(
        "iterations",
        help="Number of iterations. Default: 1",
        type=int,
        default=1,
    )
    return parser.parse_args()


def log(kws):
    print(
        "dataset: {} protected: {} frac: {} model: {} privileged: {} iteration: {}".format(
            kws["dataset_label"],
            kws["protected"],
            kws["frac"],
            kws["model"],
            kws["privileged"],
            kws["iteration"],
        )
    )


if __name__ == "__main__":
    args = parse_args()
    rows = []

    dataset_label, protected = args.dataset.split("-")
    frac = 0.6
    full = DATASET_MAP[dataset_label](
        protected_attribute_names=[protected],
        privileged_classes=PRIVILEGED_CLASSES_MAP[dataset_label][protected],
        features_to_keep=FEATURES[dataset_label],
    )
    # without distribution change
    nochange = full.split([0.75], shuffle=True)[0].split([frac], shuffle=True)[0]

    for iteration in range(0, args.iterations):
        train, test = full.split([0.75], shuffle=True)
        # with distribution change
        subset = train.split([frac], shuffle=True)[0]
        subset_nochange = nochange.split([1.0], shuffle=True)[0]
        for model in MODELS:
            for privileged in PRIVILEGED:
                if model is None:
                    row = compute_data_metrics(
                        dataset=subset,
                        dataset_label=dataset_label,
                        model="None",
                        frac=frac,
                        protected=protected,
                        privileged=privileged,
                        iteration=iteration,
                    )
                    rows.append(row)
                    log(row)

                    row = compute_data_metrics(
                        dataset=subset_nochange,
                        dataset_label=dataset_label,
                        model="None",
                        frac=frac,
                        protected=protected,
                        privileged=privileged,
                        iteration=iteration,
                        change=True,
                    )
                    rows.append(row)
                else:
                    pipe = make_pipeline(StandardScaler(), model())
                    pipe.fit(X=subset.features, y=subset.labels.ravel())
                    y_pred = pipe.predict(test.features).reshape(-1, 1)
                    classified = test.copy()
                    classified.labels = y_pred

                    row = compute_model_metrics(
                        dataset=test,
                        classified_dataset=classified,
                        dataset_label=dataset_label,
                        model=pipe.steps[-1][0],
                        frac=frac,
                        protected=protected,
                        privileged=privileged,
                        iteration=iteration,
                    )
                    rows.append(row)
                    log(row)

                    pipe = make_pipeline(StandardScaler(), model())
                    pipe.fit(X=subset_nochange.features, y=subset_nochange.labels.ravel())
                    y_pred = pipe.predict(test.features).reshape(-1, 1)
                    classified = test.copy()
                    classified.labels = y_pred

                    row = compute_model_metrics(
                        dataset=test,
                        classified_dataset=classified,
                        dataset_label=dataset_label,
                        model=pipe.steps[-1][0],
                        frac=frac,
                        protected=protected,
                        privileged=privileged,
                        iteration=iteration,
                        change=True,
                    )
                    rows.append(row)
                    log(row)

    write_csv(
        filename=os.path.join(
            DATADIR,
            "exp-distribution-change-{}-{}-{}.csv".format(
                dataset_label, protected, args.iterations
            ),
        ),
        rows=rows,
    )
