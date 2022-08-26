"""Compute & store metrics.

The data is stored in a csv where the columns consist of the metrics
along with the associated dataset, subset of dataset used (full, train
& test) and model. Missing values in metrics column imply that the
metric does not apply for that example. The `privileged' column
indicates if the metric was conditioned on the {un,}privileged_classes
(mimics the implementation of the *metric.py classes in aif360).

"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from aif360.datasets import AdultDataset, CompasDataset, BankDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
import csv
import itertools
import os

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATADIR = os.path.join(ROOTDIR, "data")
MODELS = [LogisticRegression, DecisionTreeClassifier]
PRIVILEGED = [None, True, False]


def write_csv(filename, rows):
    """Write data to csv file.

    Args:
        filename: Str, path-like name of csv file
        rows: List[Dict], data to save

    Returns:
        None

    """
    with open(filename, "w", newline="") as f:
        header = list(set(itertools.chain(*[row.keys() for row in rows])))
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def new_row(dataset, subset, model, protected, privileged):
    """Return new row.

    Args:
        dataset: Str, name of dataset
        subset: Str, name of subset
        model: Str, name of model
        protected: Str, name of protected attribute
        privileged: Str, condition on {,un}privileged group

    Returns:
        row: Dict

    """
    row = {}
    row["dataset"] = dataset
    row["subset"] = subset
    row["model"] = model
    row["protected"] = protected
    row["privileged"] = privileged

    return row


def populate_data_metrics(rows, dataset, protected, **kwargs):
    """Populate rows with data metrics.

    This method calculates the data metrics for each protected
    attribute in `protected`, for each condition in `PRIVILEGED`.

    Args:
        rows: List[Dict]
        dataset: aif360.datasets.StandardDataset
        protected: List[Str], protected attributes
        kwargs: Dict, positional arguments for new_row

    Returns:
        rows: List[Dict]

    """
    for attr in protected:
        p = [{attr: 1}]
        u = [{attr: 0}]
        metrics = BinaryLabelDatasetMetric(
            dataset=dataset, privileged_groups=p, unprivileged_groups=u
        )

        for privileged in PRIVILEGED:
            row = new_row(
                dataset=kwargs["dataset_label"],
                subset=kwargs["subset"],
                model=kwargs["model"],
                protected=attr,
                privileged=str(privileged),
            )
            row["num_positives"] = metrics.num_positives(privileged=privileged)
            row["num_negatives"] = metrics.num_negatives(privileged=privileged)
            row["base_rate"] = metrics.base_rate(privileged=privileged)
            if privileged is None:
                # TODO add the remaining metrics next
                row["disparate_impact"] = metrics.disparate_impact()
                row[
                    "statistical_parity_difference"
                ] = metrics.statistical_parity_difference()
                # row['consistency'] = metrics.consistency()
                # row['smoothed_empirical_differential_fairness'] = metrics.smoothed_empirical_differential_fairness()

            rows.append(row)

    return rows


def populate_model_metrics(rows, dataset, classified_dataset, protected, **kwargs):
    """Populate row with model metrics.

    Args:
        rows: List[Dict]
        dataset: aif360.datasets.StandardDataset
        classified_dataset: aif360.datasets.StandardDataset
        protected: List[Str], protected attributes
        kwargs: Dict, positional arguments for new_row

    Returns:
        rows: List[Dict]

    """
    for attr in protected:
        p = [{attr: 1}]
        u = [{attr: 0}]
        metrics = ClassificationMetric(
            dataset=dataset,
            classified_dataset=classified_dataset,
            privileged_groups=p,
            unprivileged_groups=u,
        )

        for privileged in PRIVILEGED:
            row = new_row(
                dataset=kwargs["dataset_label"],
                subset=kwargs["subset"],
                model=kwargs["model"],
                protected=attr,
                privileged=str(privileged),
            )
            # TODO add the rate difference, rate ratio metrics next
            # TODO add the remaining metrics after that
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

            rows.append(row)

    return rows


if __name__ == "__main__":
    rows = []
    datasets = [
        ("adult", AdultDataset, ["sex", "race"]),
        ("compas", CompasDataset, ["sex", "race"]),
        ("bank", BankDataset, ["age"]),
    ]

    for dataset_label, dataset, protected in datasets:
        full = dataset()
        train, test = full.split([0.75], shuffle=True, seed=42)

        rows = populate_data_metrics(
            rows=rows,
            dataset=full,
            protected=protected,
            dataset_label=dataset_label,
            subset="full",
            model="None",
        )

        rows = populate_data_metrics(
            rows=rows,
            dataset=train,
            protected=protected,
            dataset_label=dataset_label,
            subset="train",
            model="None",
        )

        rows = populate_data_metrics(
            rows=rows,
            dataset=test,
            protected=protected,
            dataset_label=dataset_label,
            subset="test",
            model="None",
        )

        for model in MODELS:
            pipe = make_pipeline(StandardScaler(), model())
            pipe.fit(X=train.features, y=train.labels.ravel())
            y_pred = pipe.predict(test.features).reshape(-1, 1)
            classified = test.copy()
            classified.labels = y_pred

            rows = populate_model_metrics(
                rows=rows,
                dataset=test,
                classified_dataset=classified,
                protected=protected,
                dataset_label=dataset_label,
                subset="test",
                model=pipe.steps[-1][0],
            )

    write_csv(filename=os.path.join(DATADIR, "data.csv"), rows=rows)
