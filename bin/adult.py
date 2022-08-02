"""Compute & store metrics for adult dataset.

The data is stored in a csv where the columns consist of the metrics
along with the associated dataset, subset of dataset used (full, train
& test) and model (np.nan when no model was trained). Missing values
in metrics column imply that the metric does not apply for that
example. The `privileged' column carries a Bool (True, False, np.nan)
indicating if the metric was conditioned on the
{un,}privileged_classes (mimics the implementation of the *metric.py
classes in aif360).

"""

from aif360.datasets import AdultDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import csv
import itertools
import os

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATADIR = os.path.join(ROOTDIR, "data")


def write_csv(filename, rows):
    with open(filename, "w", newline="") as f:
        header = list(set(itertools.chain(*[row.keys() for row in rows])))
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def newrow(dataset, subset, privileged, protected, model=None):
    row = {}
    row["dataset"] = dataset
    row["subset"] = subset
    row["privileged"] = str(privileged)
    row["protected"] = protected
    if model:
        row["model"] = model
    else:
        row["model"] = str(model)

    return row


def populate_data_metrics(row, metrics, condition):
    row["num_positives"] = metrics.num_positives(privileged=condition)
    row["num_negatives"] = metrics.num_negatives(privileged=condition)
    row["base_rate"] = metrics.base_rate(privileged=condition)
    if condition is None:
        # TODO add the remaining metrics next
        row["disparate_impact"] = metrics.disparate_impact()
        row["statistical_parity_difference"] = metrics.statistical_parity_difference()
        # row['consistency'] = metrics.consistency()
        # row['smoothed_empirical_differential_fairness'] = metrics.smoothed_empirical_differential_fairness()

    return row


def populate_model_metrics(row, metrics, condition):
    # TODO add the rate difference, rate ratio metrics next
    # TODO add the remaining metrics after that
    # binary confusion matrix
    row["TP"] = metrics.num_true_positives(privileged=condition)
    row["FP"] = metrics.num_false_positives(privileged=condition)
    row["FN"] = metrics.num_false_negatives(privileged=condition)
    row["TN"] = metrics.num_true_negatives(privileged=condition)
    # generalized binary confusion matrix
    row["GTP"] = metrics.num_generalized_true_positives(privileged=condition)
    row["GFP"] = metrics.num_generalized_false_positives(privileged=condition)
    row["GFN"] = metrics.num_generalized_false_negatives(privileged=condition)
    row["GTN"] = metrics.num_generalized_true_negatives(privileged=condition)
    # performance measures
    row["TPR"] = metrics.true_positive_rate(privileged=condition)
    row["FPR"] = metrics.false_positive_rate(privileged=condition)
    row["FNR"] = metrics.false_negative_rate(privileged=condition)
    row["TNR"] = metrics.true_negative_rate(privileged=condition)
    row["PPV"] = metrics.positive_predictive_value(privileged=condition)
    row["NPV"] = metrics.negative_predictive_value(privileged=condition)
    row["FDR"] = metrics.false_discovery_rate(privileged=condition)
    row["FOR"] = metrics.false_omission_rate(privileged=condition)
    row["accuracy"] = metrics.accuracy(privileged=condition)
    # generalized performance measures
    row["GTPR"] = metrics.generalized_true_positive_rate(privileged=condition)
    row["GFPR"] = metrics.generalized_false_positive_rate(privileged=condition)
    row["GFNR"] = metrics.generalized_false_negative_rate(privileged=condition)
    row["GTNR"] = metrics.generalized_true_negative_rate(privileged=condition)
    if condition is None:
        row["disparate_impact"] = metrics.disparate_impact()
        row["statistical_parity_difference"] = metrics.statistical_parity_difference()
        row["theil_index"] = metrics.theil_index()

    return row


if __name__ == "__main__":

    # TODO see docs for StandardDataset; it does a number of preprocessing
    # steps. We may want to revisit these stratergies.

    adult = AdultDataset()
    train, test = adult.split([0.75], shuffle=True, seed=42)
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(X=train.features, y=train.labels.ravel())
    y_truth = test.labels.ravel()
    y_pred = pipe.predict(test.features).reshape(
        -1, 1
    )  # ClassificationMetric expects 1D column vector
    classified = test.copy()
    classified.labels = y_pred

    protected = "sex"
    p = [{protected: 1}]
    u = [{protected: 0}]

    model_metrics = ClassificationMetric(
        dataset=test,
        classified_dataset=classified,
        privileged_groups=p,
        unprivileged_groups=u,
    )

    # TODO create product of datasets & conditions to avoid nested loop
    datasets = [("full", adult), ("train", train), ("test", test)]
    conditions = [None, True, False]

    rows = []

    # dataset metrics
    for label, dataset in datasets:
        data_metrics = BinaryLabelDatasetMetric(
            dataset=dataset, privileged_groups=p, unprivileged_groups=u
        )

        for condition in conditions:
            # model metrics
            if label == "test":
                row = newrow(
                    dataset="adult",
                    subset="test",
                    privileged=condition,
                    protected=protected,
                    model=pipe.steps[-1][0],
                )  # last element of list, first element of tuple
                row = populate_model_metrics(row, model_metrics, condition)
                rows.append(row)

            # data metrics
            row = newrow(
                dataset="adult", subset=label, privileged=condition, protected=protected
            )
            row = populate_data_metrics(row, data_metrics, condition)
            rows.append(row)

    write_csv(filename=os.path.join(DATADIR, "adult.csv"), rows=rows)
