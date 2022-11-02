"""Experiments with feature set.

This script conducts experiments with the feature set of the data. For
a given dataset, fairness metrics are calculated for varying size of
features (with a minimum of 3 features).

Usage:
The script accepts two positional arguments:
    1. name of dataset & protected attribute in data-pattr format
    2. number of iterations

E.g. the following command runs the experiment for adult-sex dataset
for 5 iterations:

    python3 src/experiments/exp-features-sets.py adult-sex 5

See python3 src/experiments/exp-feature-sets.py --help for more details.

Use the bin/exp-features-sets.bash script to execute this script for
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

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATADIR = os.path.join(ROOTDIR, "data")

# NOTE: we need the following hack since we want to execute this file
# as a script from the command line. Python by default, adds the
# directory of the script being executed into sys.path so the
# following imports don't work if we don't manipulate sys.path
# ourselves.
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
MIN_FEATURES_TO_KEEP = 3
PRIVILEGED = [None, True, False]

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


def generate_feature_sets(dataset_label, random=False):
    """Map for generating feature sets based on dataset.

    Args:
        dataset_label: Str, name of dataset
        random: Bool, defaults to False. When True randomise feature
        order.

    Returns:
        feature_sets: List[List], list of feature sets where each item
        is a list of features to keep.
    """

    feature_sets = []
    features = FEATURES[dataset_label].copy()

    # NOTE: we copy the list since python passes lists by reference (not value)!
    # TODO: refactor this using lambda & map()
    while len(features) >= MIN_FEATURES_TO_KEEP:
        feature_sets.append(features.copy())
        # TODO: to randomise, pass random index to pop
        features.pop()

    return feature_sets


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
    feature_sets = generate_feature_sets(dataset_label)

    for iteration in range(0, args.iterations):
        for features_to_keep in feature_sets:
            train, test = train_test_split(
                dataset_label=dataset_label,
                protected=protected,
                features_to_keep=features_to_keep,
            )

            for model in MODELS:
                for privileged in PRIVILEGED:
                    row = compute_metrics(
                        dataset_label=dataset_label,
                        model=model,
                        features_to_keep=features_to_keep,
                        protected=protected,
                        privileged=privileged,
                        iteration=iteration,
                        train=train,
                        test=test,
                    )
                    rows.append(row)
                    logging.info(
                        "dataset: {} protected: {} features: {} model: {} privileged: {} iteration: {}".format(
                            dataset_label,
                            protected,
                            len(features_to_keep),
                            row["model"],
                            row["privileged"],
                            iteration,
                        )
                    )

    write_csv(
        filename=os.path.join(
            DATADIR,
            "exp-feature-sets-{}-{}-{}.csv".format(
                dataset_label, protected, args.iterations
            ),
        ),
        rows=rows,
    )
