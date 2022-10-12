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
from sklearn.ensemble import RandomForestClassifier
from aif360.metrics import BinaryLabelDatasetMetric
import json
from aif360.explainers import MetricTextExplainer, MetricJSONExplainer
import lib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random
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
    featureset = originalfeatureset
    featuresubset_init = []

    writefile = open(datapath, "w")
    writefile.write(
        "trainsizeratio"
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

    trainingdatasizelist = np.arange(0.1, 1, 0.1)

    for turn in np.arange(0, 20, 1):
        seedr = random.randint(0, 100)
        print("================================================Turn:" + str(turn))

        dataset_orig_train_total, dataset_orig_test = dataset_orig.split(
            [0.8], shuffle=True, seed=seedr
        )
        # dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=False, seed=6)

        for trainsizeratio in trainingdatasizelist:
            print("training data size: " + str(trainsizeratio))
            dataset_orig_train, dataset_orig_mmm = dataset_orig_train_total.split(
                [trainsizeratio], shuffle=True, seed=seedr
            )

            numfeatures = len(dataset_orig.feature_names)
            scale_orig = StandardScaler()

            X_train_fullfeature = scale_orig.fit_transform(dataset_orig_train.features)
            y_train = dataset_orig_train.labels.ravel()
            featurecolumn = dataset_orig_train.features[:, protected_attribute_index]
            X_train = X_train_fullfeature

            metric_orig_train = BinaryLabelDatasetMetric(
                dataset_orig_train,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
            )

            unprotsubset = dataset_orig_train.features[
                np.where(dataset_orig_train.features[:, protected_attribute_index] == 1)
            ]
            protsubset = dataset_orig_train.features[
                np.where(dataset_orig_train.features[:, protected_attribute_index] == 0)
            ]

            dataset_orig_train_pred = dataset_orig_train_total.copy(deepcopy=True)
            dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

            lmod = RandomForestClassifier(max_depth=10, n_estimators=10)
            lmod.fit(X_train, y_train)

            pv = lib.get_PV_classic(lmod, X_train, y_train)

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
                + str("0")
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


def drawFig(datasetname, protectedattribute, filepath):
    newpath = filepath.replace(".csv", "_average.csv")

    readfile = open(newpath)
    lines = readfile.readlines()
    testaccu_list = []
    equaloppfairmetric_list = []
    statislist = []
    averageoddlist = []
    disparate_impactlist = []
    traindifflist = []

    divia_add_testaccu_list = []
    divia_add_equaloppfairmetric_list = []
    divia_add_statislist = []
    divia_add_averageoddlist = []
    divia_add_disparate_impactlist = []
    divia_add_traindifflist = []

    divia_sub_testaccu_list = []
    divia_sub_equaloppfairmetric_list = []
    divia_sub_statislist = []
    divia_sub_averageoddlist = []
    divia_sub_disparate_impactlist = []

    for thisline in lines:
        print(thisline)
        if "trainsizeratio" in thisline:
            continue
        splits = thisline.split(",")
        feature = splits[1]
        if feature == "1" or feature == "0":
            continue
        traindifflist.append((float(splits[3])))
        testaccu_list.append((float(splits[4])))
        equaloppfairmetric_list.append((float(splits[5])))
        statislist.append((float(splits[6])))
        averageoddlist.append((float(splits[7])))
        disparate_impactlist.append((float(splits[8])))

        divia_add_testaccu_list.append((float(splits[12]) + float(splits[4])))
        divia_add_equaloppfairmetric_list.append((float(splits[13]) + float(splits[5])))
        divia_add_statislist.append((float(splits[14]) + float(splits[6])))
        divia_add_averageoddlist.append((float(splits[15]) + float(splits[7])))
        divia_add_disparate_impactlist.append((float(splits[16]) + float(splits[8])))

        divia_sub_testaccu_list.append(-(float(splits[12]) - float(splits[4])))
        divia_sub_equaloppfairmetric_list.append(
            -(float(splits[13]) - float(splits[5]))
        )
        divia_sub_statislist.append(-(float(splits[14]) - float(splits[6])))
        divia_sub_averageoddlist.append(-(float(splits[15]) - float(splits[7])))
        divia_sub_disparate_impactlist.append(-(float(splits[16]) - float(splits[8])))

    range_ = np.arange(0.1, 1.0, 0.1)
    fig, ax1 = plt.subplots(figsize=(4, 6))

    lines += ax1.plot(
        range_, statislist, ".-.", color="b", label="statistical parity", linewidth=5
    )
    ax1.fill_between(
        range_, divia_add_statislist, divia_sub_statislist, facecolor="b", alpha=0.1
    )

    lines += ax1.plot(
        range_,
        averageoddlist,
        "--",
        color="black",
        label="average abs odds",
        linewidth=5,
    )
    ax1.fill_between(
        range_,
        divia_add_averageoddlist,
        divia_sub_averageoddlist,
        facecolor="orange",
        alpha=0.1,
    )

    lines += ax1.plot(
        range_,
        equaloppfairmetric_list,
        "-",
        marker="o",
        color="r",
        label="equal opportunity",
        linewidth=5,
        markersize=9,
    )
    ax1.fill_between(
        range_,
        divia_add_equaloppfairmetric_list,
        divia_sub_equaloppfairmetric_list,
        facecolor="r",
        alpha=0.1,
    )

    lines += ax1.plot(
        range_,
        disparate_impactlist,
        ":",
        color="green",
        label="disparate impact",
        linewidth=7,
    )
    ax1.fill_between(
        range_,
        divia_add_disparate_impactlist,
        divia_sub_disparate_impactlist,
        facecolor="green",
        alpha=0.1,
    )
    ax1.set_title(
        datasetname + " - " + protectedattribute.lower(), fontsize=25, fontweight="bold"
    )
    ax1.set_xlabel("", fontsize=28, fontweight="bold")
    ax1.set_ylabel("", color="black", fontsize=28, fontweight="bold")
    ax1.xaxis.set_tick_params(labelsize=25)
    ax1.yaxis.set_tick_params(labelsize=28)

    ax1.set_ylim((-0.1, 1.0))
    from matplotlib.ticker import FormatStrFormatter

    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    plt.xticks(np.arange(min(range_), max(range_) + 0.01, 0.2))
    plt.savefig(
        "./../plots/" + datasetname + "-" + protectedattribute + "-ds.pdf",
        bbox_inches="tight",
    )

    plt.show()


def runall():

    datasetnamelist = [["bank", "age"]]
    for i in datasetnamelist:
        filepath = "./../results/" + i[0] + "-" + i[1] + "-2d-datasize.csv"
        collectdata(i[0], i[1], filepath)
        lib.get_average(filepath)
        drawFig(i[0], i[1], filepath)


if __name__ == "__main__":
    runall()
