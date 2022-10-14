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

    python3 bin/exp-features-sets.py adult-sex 5

See python3 bin/exp-feature-sets.py --help for more details.

Use the bin/exp-features-sets.bash script to execute this script for
all datasets in parallel.

"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from aif360.datasets import (
    AdultDataset,
    CompasDataset,
    BankDataset,
    GermanDataset,
    MEPSDataset21,
)
import os
import sys
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.metrics import populate_data_metrics, populate_model_metrics
from src.csv import write_csv

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATADIR = os.path.join(ROOTDIR, "data")
MODELS = [
    LogisticRegression,
    DecisionTreeClassifier,
    AdaBoostClassifier,
    RandomForestClassifier,
]
MIN_FEATURES_TO_KEEP = 3
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
with open(os.path.join(ROOTDIR, "bin", "features.json")) as f:
    FEATURES = json.load(f)


def generate_feature_sets(dataset, random=False):
    """Map for generating feature sets based on dataset.

    Args:
        dataset: Str, name of dataset
        random: Bool, defaults to False. When True randomise feature
        order.

    Returns:
        feature_sets: List[List], list of feature sets where each item
        is a list of features to keep.
    """

    feature_sets = []
    features = FEATURES[dataset].copy()

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
    dataset = DATASET_MAP[dataset_label]
    feature_sets = generate_feature_sets(dataset_label)

    for iteration in range(0, args.iterations):
        for features_to_keep in feature_sets:
            full = dataset(
                protected_attribute_names=[protected],
                privileged_classes=PRIVILEGED_CLASSES_MAP[dataset_label][protected],
                features_to_keep=features_to_keep,
            )
            train, test = full.split([0.75], shuffle=True)

            populate_data_metrics(
                rows=rows,
                dataset=test,
                protected=protected,
                kwargs={
                    "dataset_label": dataset_label,
                    "subset": "test",
                    "model": "None",
                    "num_features": len(features_to_keep),
                },
            )
            logging.info(
                "iteration: {} dataset: {} protected: {} features: {} model: None".format(
                    iteration, dataset_label, protected, len(features_to_keep)
                )
            )

            for model in MODELS:
                pipe = make_pipeline(StandardScaler(), model())
                pipe.fit(X=train.features, y=train.labels.ravel())
                y_pred = pipe.predict(test.features).reshape(-1, 1)
                classified = test.copy()
                classified.labels = y_pred

                populate_model_metrics(
                    rows=rows,
                    dataset=test,
                    classified_dataset=classified,
                    protected=protected,
                    kwargs={
                        "dataset_label": dataset_label,
                        "subset": "test",
                        "model": pipe.steps[-1][0],
                        "num_features": len(features_to_keep),
                    },
                )

                logging.info(
                    "iteration: {} dataset: {} protected: {} features: {} model: {}".format(
                        iteration,
                        dataset_label,
                        protected,
                        len(features_to_keep),
                        pipe.steps[-1][0],
                    )
                )

    write_csv(
        filename=os.path.join(
            DATADIR, "exp-feature-sets-{}-{}.csv".format(dataset_label, protected)
        ),
        rows=rows,
    )
