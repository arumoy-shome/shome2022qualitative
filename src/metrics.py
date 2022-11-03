"""Module for calculating fairness metrics.

This provides functions to calculate the fairness metrics for a given
dataset & ML model.
"""

from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric


def new_row(kwargs):
    """Return new row.

    Args:
        kwargs: Dict

    Returns:
        row: Dict

    """
    row = {}
    for k, v in kwargs.items():
        row[k] = v

    return row


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
        iteration: Int

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

    return new_row(kwargs)


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
        iteration: Int

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

    return new_row(kwargs)
