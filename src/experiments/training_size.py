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

    python3 src/experiments/training_size.py adult-sex 5

See python3 src/experiments/training_size.py --help for more details.

Use the bin/exp-training-size.bash script to execute this script for
all datasets in parallel.

"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import os
import sys
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATADIR = os.path.join(ROOTDIR, "data")

sys.path.insert(0, ROOTDIR)
from src.metics import compute_metrics
from src.csv import write_csv

MODELS = [
    None,
    LogisticRegression,
    DecisionTreeClassifier,
    AdaBoostClassifier,
    RandomForestClassifier,
]
PRIVILEGED = [None, True, False]


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


if __name__ == "__main__":
    args = parse_args()
    rows = []

    dataset_label, protected = args.dataset.split("-")
