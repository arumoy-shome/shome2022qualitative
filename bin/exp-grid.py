"""Experiments with feature & training set.

This script conducts experiments with the feature & training set of
the data. For a given dataset, fairness metrics are calculated for
varying training & feature size.

Usage:
The script accepts two positional arguments:
    1. name of dataset & protected attribute in data-pattr format
    2. number of iterations


E.g. the following command runs the experiment for adult-sex dataset
for 5 iterations:

    python3 bin/exp-grid.py adult-sex 5

See python3 bin/exp-grid.py --help for more details.

Use the bin/exp-grid.bash script to execute this script for all
datasets in parallel.

"""
from src.utils import write_csv
from src.metrics import compute_data_metrics, compute_model_metrics
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
import numpy as np
import os
import sys
import json
import argparse
import logging
import random

logging.basicConfig(level=logging.INFO)

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATADIR = os.path.join(ROOTDIR, "data")

# NOTE: we need the following hack since we want to execute this file
# as a script from the command line. Python by default, adds the
# directory of the script being executed into sys.path so the
# following imports don't work if we don't manipulate sys.path
# ourselves.
sys.path.insert(0, ROOTDIR)

MODELS = [
    None,
    LogisticRegression,
    DecisionTreeClassifier,
    AdaBoostClassifier,
    RandomForestClassifier,
]
MIN_FEATURES_TO_KEEP = 3
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

# NOTE: the following features are derived from the corresponding
# class in the aif360.datasets module. The default set & order
# provided by aif360 is used. The follow strategy is used to derive
# the list of features:
# 1) use value of `features_to_keep` argument if it is not empty else
# 2) use value of `column_names` passed to `pd.read_csv` function else
# 3) manually check the csv file/data documentation to obtain list of
#    features.
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
        f"dataset: {kws['dataset_label']} protected: {kws['protected']} features: {kws['num_features']} frac: {kws['frac']} model: {kws['model']} privileged: {kws['privileged']} iteration: {kws['iteration']}"
    )


if __name__ == "__main__":
    args = parse_args()
    rows = []

    dataset_label, protected = args.dataset.split("-")

    for iteration in range(0, args.iterations):
        for num_features in range(MIN_FEATURES_TO_KEEP, len(FEATURES[dataset_label]) + 1):
            for frac in np.linspace(start=0.1, stop=1.0, num=10):
                features_to_keep = random.sample(
                    FEATURES[dataset_label], num_features)
                full = DATASET_MAP[dataset_label](
                    protected_attribute_names=[protected],
                    privileged_classes=PRIVILEGED_CLASSES_MAP[dataset_label][protected],
                    features_to_keep=features_to_keep,
                )
                train, test = full.split([0.75], shuffle=True)
                subset, _ = train.split([frac], shuffle=True)
                for model in MODELS:
                    for privileged in PRIVILEGED:
                        if model is None:
                            row = compute_data_metrics(
                                dataset=subset,
                                dataset_label=dataset_label,
                                model="None",
                                frac=frac,
                                num_features=num_features,
                                protected=protected,
                                privileged=privileged,
                                iteration=iteration,
                            )
                            rows.append(row)
                            log(row)
                        else:
                            pipe = make_pipeline(StandardScaler(), model())
                            pipe.fit(X=subset.features,
                                     y=subset.labels.ravel())
                            y_pred = pipe.predict(test.features).reshape(-1, 1)
                            classified = test.copy()
                            classified.labels = y_pred

                            row = compute_model_metrics(
                                dataset=test,
                                classified_dataset=classified,
                                dataset_label=dataset_label,
                                model=pipe.steps[-1][0],
                                frac=frac,
                                num_features=num_features,
                                protected=protected,
                                privileged=privileged,
                                iteration=iteration,
                            )
                            rows.append(row)
                            log(row)

    write_csv(
        filename=os.path.join(
            DATADIR,
            "exp-grid-{}-{}-{}.csv".format(
                dataset_label, protected, args.iterations
            ),
        ),
        rows=rows,
    )
