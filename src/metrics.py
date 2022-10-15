"""Module for calculating fairness metrics.

This provides functions to calculate the fairness metrics for a given
dataset & ML model.
"""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import (
    AdultDataset,
    CompasDataset,
    BankDataset,
    GermanDataset,
    MEPSDataset21,
)
from . import csv

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

memory = {}


def train_test_split(dataset_label, protected, features_to_keep):
    if dataset_label not in memory.keys():
        # top level key does not exists
        # compute & cache everything
        memory[dataset_label] = {}
    else:
        # top level key exists
        # bottom level key may not exist
        if len(features_to_keep) not in memory[dataset_label]:
            # bottom level key also does not exist
            # compute & cache everything
            pass
        else:
            # result already cached
            return memory[dataset_label][len(features_to_keep)]

    dataset = DATASET_MAP[dataset_label]
    full = dataset(
        protected_attribute_names=[protected],
        privileged_classes=PRIVILEGED_CLASSES_MAP[dataset_label][protected],
        features_to_keep=features_to_keep,
    )
    memory[dataset_label][len(features_to_keep)] = full.split([0.75], shuffle=True)

    return memory[dataset_label][len(features_to_keep)]


def compute_metrics(dataset_label, model, features_to_keep, protected, privileged):
    """Map for populating data or model metrics.

    In:
        dataset_label: Str
        model: Obj, sklearn ML model
        features_to_keep: List, list of features to use in training data
        protected: Str
        privileged: None or Bool

    Returns:
        metrics: Dict
    """

    train, test = train_test_split(
        dataset_label=dataset_label,
        protected=protected,
        features_to_keep=features_to_keep,
    )

    if model is None:
        return compute_data_metrics(
            dataset=test,
            dataset_label=dataset_label,
            model="None",
            num_features=len(features_to_keep),
            protected=protected,
            privileged=privileged,
        )

    else:
        pipe = make_pipeline(StandardScaler(), model())
        pipe.fit(X=train.features, y=train.labels.ravel())
        y_pred = pipe.predict(test.features).reshape(-1, 1)
        classified = test.copy()
        classified.labels = y_pred

        return compute_model_metrics(
            dataset=test,
            classified_dataset=classified,
            dataset_label=dataset_label,
            model=pipe.steps[-1][0],
            num_features=len(features_to_keep),
            protected=protected,
            privileged=privileged,
        )


def compute_data_metrics(**kwargs):
    """Compute data metrics.

    This method calculates the data metrics for the protected
    attribute, for each condition in `PRIVILEGED`.

    Args:
        dataset: aif360.datasets.StandardDataset
        dataset_label: Str
        model: Str
        num_features: Int
        protected: Str, protected attribute
        privileged: None or Bool

    Returns:
        Metrics: Dict

    """

    p = [{kwargs["protected"]: 1}]
    u = [{kwargs["protected"]: 0}]
    metrics = BinaryLabelDatasetMetric(
        dataset=kwargs.pop("dataset"), privileged_groups=p, unprivileged_groups=u
    )

    kwargs["num_positives"] = metrics.num_positives(privileged=kwargs["privileged"])
    kwargs["num_negatives"] = metrics.num_negatives(privileged=kwargs["privileged"])
    kwargs["base_rate"] = metrics.base_rate(privileged=kwargs["privileged"])
    if kwargs["privileged"] is None:
        kwargs["disparate_impact"] = metrics.disparate_impact()
        kwargs[
            "statistical_parity_difference"
        ] = metrics.statistical_parity_difference()
    kwargs["privileged"] = str(kwargs["privileged"])

    return csv.new_row(kwargs)


def compute_model_metrics(**kwargs):
    """Compute model metrics.

    Args:
        dataset: aif360.datasets.StandardDataset
        classified_dataset: aif360.datasets.StandardDataset
        dataset_label: Str
        model: Str
        num_features: Int
        protected: Str, protected attribute
        privileged: None or Bool

    Returns:
        Metrics: Dict

    """

    p = [{kwargs["protected"]: 1}]
    u = [{kwargs["protected"]: 0}]
    metrics = ClassificationMetric(
        dataset=kwargs.pop("dataset"),
        classified_dataset=kwargs.pop("classified_dataset"),
        privileged_groups=p,
        unprivileged_groups=u,
    )

    # binary confusion matrix
    kwargs["TP"] = metrics.num_true_positives(privileged=kwargs["privileged"])
    kwargs["FP"] = metrics.num_false_positives(privileged=kwargs["privileged"])
    kwargs["FN"] = metrics.num_false_negatives(privileged=kwargs["privileged"])
    kwargs["TN"] = metrics.num_true_negatives(privileged=kwargs["privileged"])
    # performance measures
    recall = metrics.true_positive_rate(privileged=kwargs["privileged"])  # alias recall
    precision = metrics.positive_predictive_value(
        privileged=kwargs["privileged"]
    )  # alias precision
    kwargs["TPR"] = recall
    kwargs["FPR"] = metrics.false_positive_rate(privileged=kwargs["privileged"])
    kwargs["FNR"] = metrics.false_negative_rate(privileged=kwargs["privileged"])
    kwargs["TNR"] = metrics.true_negative_rate(privileged=kwargs["privileged"])
    kwargs["PPV"] = precision
    kwargs["accuracy"] = metrics.accuracy(privileged=kwargs["privileged"])
    kwargs["f1"] = (2 * precision * recall) / (
        precision + recall
    )  # harmonic mean of precision & recall
    # generalized performance measures
    if kwargs["privileged"] is None:
        kwargs["disparate_impact"] = metrics.disparate_impact()
        kwargs[
            "statistical_parity_difference"
        ] = metrics.statistical_parity_difference()
        kwargs["theil_index"] = metrics.theil_index()
        kwargs["average_abs_odds_difference"] = metrics.average_abs_odds_difference()
        kwargs[
            "true_positive_rate_difference"
        ] = (
            metrics.true_positive_rate_difference()
        )  # alias equal_opportunity_difference
    kwargs["privileged"] = str(kwargs["privileged"])

    return csv.new_row(kwargs)
