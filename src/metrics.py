"""Module for calculating fairness metrics.

This provides functions to calculate the fairness metrics for a given
dataset & ML model.
"""

from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from . import csv


PRIVILEGED = [None, True, False]


def populate_data_metrics(rows, dataset, protected, kwargs):
    """Populate rows with data metrics.

    This method calculates the data metrics for the protected
    attribute, for each condition in `PRIVILEGED`.

    Args:
        rows: List[Dict]
        dataset: aif360.datasets.StandardDataset
        protected: Str, protected attribute
        kwargs: Dict, positional arguments for new_row

    Returns:
        rows: List[Dict]

    """

    p = [{protected: 1}]
    u = [{protected: 0}]
    metrics = BinaryLabelDatasetMetric(
        dataset=dataset, privileged_groups=p, unprivileged_groups=u
    )

    for privileged in PRIVILEGED:
        kwargs["protected"] = protected
        kwargs["privileged"] = str(privileged)
        row = csv.new_row(
            kwargs
        )  # break the pass-by-reference; we want new dict every time
        row["num_positives"] = metrics.num_positives(privileged=privileged)
        row["num_negatives"] = metrics.num_negatives(privileged=privileged)
        row["base_rate"] = metrics.base_rate(privileged=privileged)
        if privileged is None:
            row["disparate_impact"] = metrics.disparate_impact()
            row[
                "statistical_parity_difference"
            ] = metrics.statistical_parity_difference()
            # NOTE there are other metrics as well which I did not investigate
            # row['consistency'] = metrics.consistency()
            # row['smoothed_empirical_differential_fairness'] = metrics.smoothed_empirical_differential_fairness()

        rows.append(row)

    return rows


def populate_model_metrics(rows, dataset, classified_dataset, protected, kwargs):
    """Populate row with model metrics.

    Args:
        rows: List[Dict]
        dataset: aif360.datasets.StandardDataset
        classified_dataset: aif360.datasets.StandardDataset
        protected: Str, protected attribute
        kwargs: Dict, positional arguments for new_row

    Returns:
        rows: List[Dict]

    """

    p = [{protected: 1}]
    u = [{protected: 0}]
    metrics = ClassificationMetric(
        dataset=dataset,
        classified_dataset=classified_dataset,
        privileged_groups=p,
        unprivileged_groups=u,
    )

    for privileged in PRIVILEGED:
        kwargs["protected"] = protected
        kwargs["privileged"] = str(privileged)
        row = csv.new_row(kwargs)
        # binary confusion matrix
        row["TP"] = metrics.num_true_positives(privileged=privileged)
        row["FP"] = metrics.num_false_positives(privileged=privileged)
        row["FN"] = metrics.num_false_negatives(privileged=privileged)
        row["TN"] = metrics.num_true_negatives(privileged=privileged)
        # generalized binary confusion matrix
        # row["GTP"] = metrics.num_generalized_true_positives(privileged=privileged)
        # row["GFP"] = metrics.num_generalized_false_positives(privileged=privileged)
        # row["GFN"] = metrics.num_generalized_false_negatives(privileged=privileged)
        # row["GTN"] = metrics.num_generalized_true_negatives(privileged=privileged)
        # performance measures
        recall = metrics.true_positive_rate(privileged=privileged)  # alias recall
        precision = metrics.positive_predictive_value(
            privileged=privileged
        )  # alias precision
        row["TPR"] = recall
        row["FPR"] = metrics.false_positive_rate(privileged=privileged)
        row["FNR"] = metrics.false_negative_rate(privileged=privileged)
        row["TNR"] = metrics.true_negative_rate(privileged=privileged)
        row["PPV"] = precision
        # row["NPV"] = metrics.negative_predictive_value(privileged=privileged)
        # row["FDR"] = metrics.false_discovery_rate(privileged=privileged)
        # row["FOR"] = metrics.false_omission_rate(privileged=privileged)
        row["accuracy"] = metrics.accuracy(privileged=privileged)
        row["f1"] = (2 * precision * recall) / (
            precision + recall
        )  # harmonic mean of precision & recall
        # generalized performance measures
        # row["GTPR"] = metrics.generalized_true_positive_rate(privileged=privileged)
        # row["GFPR"] = metrics.generalized_false_positive_rate(privileged=privileged)
        # row["GFNR"] = metrics.generalized_false_negative_rate(privileged=privileged)
        # row["GTNR"] = metrics.generalized_true_negative_rate(privileged=privileged)
        if privileged is None:
            row["disparate_impact"] = metrics.disparate_impact()
            row[
                "statistical_parity_difference"
            ] = metrics.statistical_parity_difference()
            row["theil_index"] = metrics.theil_index()
            row["average_abs_odds_difference"] = metrics.average_abs_odds_difference()
            row[
                "true_positive_rate_difference"
            ] = (
                metrics.true_positive_rate_difference()
            )  # alias equal_opportunity_difference

        rows.append(row)

    return rows
