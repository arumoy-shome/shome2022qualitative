# Load all necessary packages
import statistics
import sys
import numpy as np
import pandas
import pandas as pd
from sklearn.feature_selection import SelectKBest
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import (
    CalibratedEqOddsPostprocessing,
)

sys.path.append("../")
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.datasets import (
    AdultDataset,
    GermanDataset,
    CompasDataset,
    BankDataset,
    MEPSDataset19,
)
from aif360.metrics import BinaryLabelDatasetMetric
from sklearn.feature_selection import chi2
from aif360.metrics import ClassificationMetric
from sklearn import tree
import json
from aif360.explainers import MetricTextExplainer, MetricJSONExplainer
import lib
from aif360.algorithms.preprocessing.reweighing import Reweighing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, accuracy_score
import random

## import dataset

from collections import OrderedDict
from aif360.metrics import ClassificationMetric


def compute_metrics(
    dataset_true, dataset_pred, unprivileged_groups, privileged_groups, disp=True
):
    """Compute the key metrics"""
    classified_metric_pred = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = 0.5 * (
        classified_metric_pred.true_positive_rate()
        + classified_metric_pred.true_negative_rate()
    )
    metrics[
        "Statistical parity difference"
    ] = classified_metric_pred.statistical_parity_difference()
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics[
        "Average odds difference"
    ] = classified_metric_pred.average_odds_difference()
    metrics[
        "Equal opportunity difference"
    ] = classified_metric_pred.equal_opportunity_difference()
    metrics["Theil index"] = classified_metric_pred.theil_index()

    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))

    return metrics


def collectdata(datasetname, protectedattribute, datapath):
    dataset_orig, privileged_groups, unprivileged_groups = lib.get_data(
        datasetname, protectedattribute
    )
    print(unprivileged_groups)

    originalfeatureset = dataset_orig.feature_names
    featureset = originalfeatureset[:]
    featuresubset_init = []

    writefile = open(datapath, "w")
    writefile.write(
        "datasetname"
        + ","
        + "turn"
        + ","
        + "trainsizeratio"
        + ","
        + "featurenum"
        + ","
        + "depth"
        + ","
        + "train.mean_difference"
        + ","
        + "testpred.accuracy"
        + ","
        + "testpred.equal_opportunity_difference"
        + ","
        + "testpred.statistical_parity_difference"
        + ","
        + "testpred.average_abs_odds_difference"
        + ","
        + "testpred.disparate_impact"
        + "\n"
    )
    writefile.close

    protected_attribute_names = [protectedattribute]
    protected_attribute_index = -1
    for each in protected_attribute_names:
        protected_attribute_index = featureset.index(each)
        featuresubset_init.append(featureset.index(each))
    for each in protected_attribute_names:
        featureset.remove(each)
    depthlist = [10]
    trainingdatasizelist = np.arange(0.1, 1, 0.1)

    for turn in np.arange(48, 50, 1):
        seedr = random.randint(0, 100)
        print("================================================Turn:" + str(turn))

        dataset_orig_train_total, dataset_orig_test = dataset_orig.split(
            [0.8], shuffle=True, seed=seedr
        )

        for trainsizeratio in trainingdatasizelist:
            print("training data size: " + str(trainsizeratio))
            dataset_orig_train, dataset_orig_mmm = dataset_orig_train_total.split(
                [trainsizeratio], shuffle=True, seed=seedr
            )

            numfeatures = len(dataset_orig.feature_names)
            scale_orig = StandardScaler()
            RW = Reweighing(
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
            )
            dataset_transf_train = RW.fit_transform(dataset_orig_train)

            X_train_fullfeature = scale_orig.fit_transform(
                dataset_transf_train.features
            )
            y_train = dataset_transf_train.labels.ravel()
            featurecolumn = dataset_transf_train.features[:, protected_attribute_index]
            X_train = X_train_fullfeature

            metric_orig_train = BinaryLabelDatasetMetric(
                dataset_transf_train,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
            )

            unprotsubset = dataset_orig_train.features[
                np.where(
                    dataset_transf_train.features[:, protected_attribute_index] == 1
                )
            ]
            protsubset = dataset_orig_train.features[
                np.where(
                    dataset_transf_train.features[:, protected_attribute_index] == 0
                )
            ]

            dataset_orig_train_pred = dataset_orig_train_total.copy(deepcopy=True)
            dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

            for depth in depthlist:

                lmod = tree.DecisionTreeClassifier(max_depth=depth)
                lmod.fit(
                    X_train,
                    y_train,
                    sample_weight=dataset_transf_train.instance_weights,
                )

                fav_idx = np.where(
                    lmod.classes_ == dataset_orig_train_total.favorable_label
                )[0][0]
                y_train_pred_prob = lmod.predict_proba(X_train)[:, fav_idx]

                X_test_fullfeature = scale_orig.transform(dataset_orig_test.features)
                X_test = X_test_fullfeature
                y_test_pred_prob = lmod.predict_proba(X_test)[:, fav_idx]

                class_thresh = 0.5
                dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1, 1)
                dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1, 1)

                y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
                y_test_pred[
                    y_test_pred_prob >= class_thresh
                ] = dataset_orig_test_pred.favorable_label
                y_test_pred[
                    ~(y_test_pred_prob >= class_thresh)
                ] = dataset_orig_test_pred.unfavorable_label
                dataset_orig_test_pred.labels = y_test_pred

                cm_transf_test = ClassificationMetric(
                    dataset_orig_test,
                    dataset_orig_test_pred,
                    unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups,
                )

                writefile = open(datapath, "a")
                writefile.write(
                    datasetname
                    + ","
                    + str(turn)
                    + ","
                    + str(trainsizeratio)
                    + ","
                    + str(numfeatures)
                    + ","
                    + str(depth)
                    + ","
                    + str(metric_orig_train.disparate_impact())
                    + ","
                    + str(1 - cm_transf_test.error_rate())
                    + ","  # 6
                    + str(cm_transf_test.equal_opportunity_difference())
                    + ","
                    + str(cm_transf_test.statistical_parity_difference())
                    + ","
                    + str(cm_transf_test.average_abs_odds_difference())
                    + ","
                    + str(cm_transf_test.disparate_impact())
                    + "\n"
                )
                writefile.close()


def format_json(json_str):
    return json.dumps(json.loads(json_str, object_pairs_hook=OrderedDict), indent=2)


def runall():
    datasetnamelist = [
        ["bank", "age"],
        ["meps", "RACE"],
        ["german", "age"],
        ["german", "sex"],
        ["compas", "race"],
        ["compas", "sex"],
    ]
    for i in datasetnamelist:
        filepath = "./newresults/" + i[0] + "-" + i[1] + "-2d-datasize-reweigh.csv"
        collectdata(i[0], i[1], filepath)
        lib.get_average(filepath)
        lib.drawFig(i[0], i[1], filepath)


if __name__ == "__main__":

    runall()
