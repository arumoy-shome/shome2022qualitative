"""Experiments with training size.

This script conducts experiments with the training size. For the a
given dataset, fairness metrics are calculated for varying size of
training data (with a minimum of 10% of original training data).

Usage:
The script accepts two positional arguments:
    1. name of dataset & protected attribute in data-pattr format
    2. number of iterations

E.g. the following command runs the experiment for adult-sex dataset
for 5 iterations:

    python3 bin/training_size.py adult-sex 5

See python3 bin/training_size.py --help for more details.

Use the bin/exp-training-size.bash script to execute this script for
all datasets in parallel.

"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import numpy as np
import os
import sys
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATADIR = os.path.join(ROOTDIR, "data")

sys.path.insert(0, ROOTDIR)
from src.metrics import compute_metrics
from src.utils import train_test_split, write_csv

MODELS = [
    None,
    LogisticRegression,
    DecisionTreeClassifier,
    AdaBoostClassifier,
    RandomForestClassifier,
]
PRIVILEGED = [None, True, False]
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


def generate_training_sets(train, random=False):
    training_sets = []

    for frac in np.arange(0.1, 1.0, 0.05):
        subset, _ = train.split(
            [frac], shuffle=False
        )  # TODO do we want to shuffle here?
        training_sets.append((frac, subset))

    training_sets.append((1.0, train))  # add the full set as well
    return training_sets


if __name__ == "__main__":
    args = parse_args()
    rows = []

    dataset_label, protected = args.dataset.split("-")

    for iteration in range(0, args.iterations):
        train, test = train_test_split(
            dataset_label=dataset_label,
            protected=protected,
            features_to_keep=FEATURES[dataset_label],
        )
        training_sets = generate_training_sets(train)
        for frac, subset in training_sets:
            for model in MODELS:
                for privileged in PRIVILEGED:
                    row = compute_metrics(
                        dataset_label=dataset_label,
                        model=model,
                        features_to_keep=FEATURES[dataset_label],
                        protected=protected,
                        privileged=privileged,
                        iteration=iteration,
                        frac=frac,
                        train=subset,
                        test=test,
                    )
                    rows.append(row)
                    logging.info(
                        "dataset: {} protected: {} frac: {:.2f} model: {} privileged: {} iteration: {}".format(
                            dataset_label,
                            protected,
                            frac,
                            row["model"],
                            row["privileged"],
                            iteration,
                        )
                    )

    write_csv(
        filename=os.path.join(
            DATADIR,
            "exp-training-sets-{}-{}-{}.csv".format(
                dataset_label, protected, args.iterations
            ),
        ),
        rows=rows,
    )
