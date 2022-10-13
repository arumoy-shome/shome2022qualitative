"""Experiments with feature set.

This script conducts experiments with the feature set of the data. For
a given dataset, fairness metrics are calculated for varying size of
features (with a minimum of 3 features).

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
from multiprocessing import Process, Manager
import os
import sys
import json
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


def fit_predict(model, train, test, classified_datasets):
    """Train & test a model with the given data.

    Args:
        model: Obj, an instance of a model from sklearn
        train: Array like, the training data
        test: Array like, the test data with actual labels (y)
        classified_datasets: Dict, shared variable between multiple
        processes used to collect the results of this function.

    Returns: None
    """

    logging.info("computing metrics for model: {}".format(model))
    pipe = make_pipeline(StandardScaler(), model())
    pipe.fit(X=train.features, y=train.labels.ravel())
    y_pred = pipe.predict(test.features).reshape(-1, 1)
    classified = test.copy()
    classified.labels = y_pred
    classified_datasets[pipe.steps[-1][0]] = classified


if __name__ == "__main__":
    iterations = sys.argv[1] if len(sys.argv) > 1 else 1
    rows = []
    datasets = [
        ("adult", AdultDataset, "sex"),
        ("adult", AdultDataset, "race"),
        ("compas", CompasDataset, "sex"),
        ("compas", CompasDataset, "race"),
        ("bank", BankDataset, "age"),
        ("german", GermanDataset, "sex"),
        ("german", GermanDataset, "age"),
        ("meps", MEPSDataset21, "RACE"),
    ]

    for iteration in range(0, int(iterations)):
        logging.info("iteration {}".format(iteration))
        for dataset_label, dataset, protected in datasets:
            logging.info(
                "computing metrics for dataset: {} protected: {}".format(
                    dataset_label, protected
                )
            )
            feature_sets = generate_feature_sets(dataset_label)

            for features_to_keep in feature_sets:
                full = dataset(
                    protected_attribute_names=[protected],
                    privileged_classes=PRIVILEGED_CLASSES_MAP[dataset_label][protected],
                    features_to_keep=features_to_keep,
                )
                train, test = full.split([0.75], shuffle=True)

                rows = populate_data_metrics(
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

                manager = Manager()
                classified_datasets = manager.dict()
                jobs = []
                for model in MODELS:
                    p = Process(
                        target=fit_predict,
                        args=(model, train, test, classified_datasets),
                    )
                    jobs.append(p)
                    p.start()

                for job in jobs:
                    job.join()

                for model, classified_dataset in classified_datasets.items():
                    rows = populate_model_metrics(
                        rows=rows,
                        dataset=test,
                        classified_dataset=classified_dataset,
                        protected=protected,
                        kwargs={
                            "dataset_label": dataset_label,
                            "subset": "test",
                            "model": model,
                            "num_features": len(features_to_keep),
                        },
                    )

    write_csv(filename=os.path.join(DATADIR, "exp-feature-sets.csv"), rows=rows)
