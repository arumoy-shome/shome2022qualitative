# Load all necessary packages
import sys
import numpy as np
import pandas
import pandas as pd
from sklearn.feature_selection import SelectKBest
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import (
    CalibratedEqOddsPostprocessing,
)

sys.path.append("../")
from sklearn.tree import DecisionTreeClassifier
from aif360.metrics import BinaryLabelDatasetMetric, SampleDistortionMetric
import statistics
import json
from aif360.explainers import MetricTextExplainer, MetricJSONExplainer

import lib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, accuracy_score
import random
from sklearn.model_selection import GridSearchCV
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


def get_actual_featurenum(featurenameslist):

    totalist = []
    for feature in featurenameslist:
        feature = feature.split("=")[0]
        if feature not in totalist:
            totalist.append(feature)

    return len(totalist)


def collectdata(datasetname, protectedattribute, datapath):
    dataset_orig, privileged_groups, unprivileged_groups = lib.get_data(
        datasetname, protectedattribute
    )
    trainsizeratio = 1.0

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
        + ","
        + "\n"
    )

    protected_attribute_names = [protectedattribute]
    protected_attribute_index = -1
    for each in protected_attribute_names:
        protected_attribute_index = featureset.index(each)
        featuresubset_init.append(featureset.index(each))
    for each in protected_attribute_names:
        featureset.remove(each)

    actualfeaturenum = get_actual_featurenum(dataset_orig.feature_names) + 1
    print("actual total feature  num")
    print(actualfeaturenum)
    featurenumlist = np.arange(1, actualfeaturenum, 1)

    depthlist = [10]

    print("total number of features:")
    print(len(dataset_orig.feature_names))

    for turn in np.arange(0, 20, 1):
        seedr = random.randint(0, 100)
        print("================================================Turn:" + str(turn))

        dataset_orig_train_total, dataset_orig_test = dataset_orig.split(
            [0.8], shuffle=True, seed=seedr
        )
        # dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=False, seed=6)
        dataset_orig_train_pred = dataset_orig_train_total.copy(deepcopy=True)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        dataset_orig_train, dataset_orig_mmm = dataset_orig_train_total.split(
            [trainsizeratio], shuffle=True, seed=seedr
        )

        for numfeatures in featurenumlist:
            print("num of features: " + str(numfeatures))
            featuresubset = list(np.copy(featuresubset_init))

            coveredfeaturelist = featuresubset[:]

            for feature in featureset:
                if len(coveredfeaturelist) == numfeatures:
                    break
                thisfeaturestring = feature.split("=")[0]
                if thisfeaturestring not in coveredfeaturelist:
                    coveredfeaturelist.append(thisfeaturestring)
                featuresubset.append(originalfeatureset.index(feature))
            scale_orig = StandardScaler()
            featuresubset = list(set(featuresubset))
            print(featuresubset)

            X_train_fullfeature = scale_orig.fit_transform(dataset_orig_train.features)
            y_train = dataset_orig_train.labels.ravel()
            X_train = X_train_fullfeature[:, featuresubset]

            metric_orig_train = BinaryLabelDatasetMetric(
                dataset_orig_train,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
            )

            for depth in depthlist:
                params = {"max_depth": list(range(4, 10))}
                lmod = GridSearchCV(
                    DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3
                )
                lmod.fit(X_train, y_train)
                fav_idx = np.where(
                    lmod.classes_ == dataset_orig_train_total.favorable_label
                )[0][0]
                y_train_pred_prob = lmod.predict_proba(X_train)[:, fav_idx]

                X_test_fullfeature = scale_orig.transform(dataset_orig_test.features)
                X_test = X_test_fullfeature[:, featuresubset]
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
                    + str(metric_orig_train.mean_difference())
                    + ","
                    + str(cm_transf_test.accuracy())
                    + ","  # 6
                    + str(cm_transf_test.equal_opportunity_difference())
                    + ","
                    + str(cm_transf_test.statistical_parity_difference())
                    + ","
                    + str(cm_transf_test.average_abs_odds_difference())
                    + ","
                    + str(cm_transf_test.disparate_impact())
                    + +"\n"
                )
    writefile.close()


def format_json(json_str):
    return json.dumps(json.loads(json_str, object_pairs_hook=OrderedDict), indent=2)


def get_average(filepath):
    readfile = open(filepath)
    lines = readfile.readlines()
    newfilepath = filepath.replace(".csv", "_average.csv")
    writefile = open(newfilepath, "w")
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
        + ","  # 9
        + "\n"  # 10
    )
    dic_string_meandiff = {}
    dic_string_accuracy = {}
    dic_string_equopp = {}
    dic_string_statis = {}
    dic_string_averageodd = {}
    dic_string_disparate = {}

    for thisline in lines:

        if "datasetname" in thisline:
            continue
        splits = thisline.split(",")
        featurenum = splits[3]
        if featurenum == "1" or featurenum == "2":
            continue
        threestring = splits[2] + "," + splits[3] + "," + splits[4]
        if threestring in dic_string_meandiff:
            dic_string_meandiff[threestring].append(abs(float(splits[5])))
            dic_string_accuracy[threestring].append(abs(float(splits[6])))
            dic_string_equopp[threestring].append(abs(float(splits[7])))
            dic_string_statis[threestring].append(abs(float(splits[8])))
            dic_string_averageodd[threestring].append(abs(float(splits[9])))
            dic_string_disparate[threestring].append(abs(1 - float(splits[10])))

        else:

            dic_string_meandiff[threestring] = [abs(float(splits[5]))]
            dic_string_accuracy[threestring] = [abs(float(splits[6]))]
            dic_string_equopp[threestring] = [abs(float(splits[7]))]
            dic_string_statis[threestring] = [abs(float(splits[8]))]
            dic_string_averageodd[threestring] = [abs(float(splits[9]))]
            dic_string_disparate[threestring] = [abs(1 - float(splits[10]))]

    for each in dic_string_meandiff:

        writefile.write(
            each
            + ","
            + str(1.0 * sum(dic_string_meandiff[each]) / len(dic_string_meandiff[each]))
            + ","
            + str(1.0 * sum(dic_string_accuracy[each]) / len(dic_string_accuracy[each]))
            + ","
            + str(1.0 * sum(dic_string_equopp[each]) / len(dic_string_equopp[each]))
            + ","
            + str(1.0 * sum(dic_string_statis[each]) / len(dic_string_statis[each]))
            + ","
            + str(
                1.0
                * sum(dic_string_averageodd[each])
                / len(dic_string_averageodd[each])
            )
            + ","
            + str(
                1.0 * sum(dic_string_disparate[each]) / len(dic_string_disparate[each])
            )
            + ","
            + str(statistics.stdev(dic_string_meandiff[each]))
            + ","
            + str(statistics.stdev(dic_string_accuracy[each]))
            + ","
            + str(statistics.stdev(dic_string_equopp[each]))
            + ","
            + str(statistics.stdev(dic_string_statis[each]))
            + ","
            + str(statistics.stdev(dic_string_averageodd[each]))
            + ","
            + str(statistics.stdev(dic_string_disparate[each]))
            + ","
            "\n"
        )
    writefile.close()


def drawFig(datasetname, protectedattribute, filepath):
    newpath = filepath.replace(".csv", "_average.csv")

    readfile = open(newpath)
    lines = readfile.readlines()
    metric_list = []
    equaloppfairmetric_list = []
    statislist = []
    averageoddlist = []
    disparate_impactlist = []
    traindifflist = []

    divia_add_metric_list = []
    divia_add_equaloppfairmetric_list = []
    divia_add_statislist = []
    divia_add_averageoddlist = []
    divia_add_disparate_impactlist = []
    divia_add_traindifflist = []

    divia_sub_metric_list = []
    divia_sub_equaloppfairmetric_list = []
    divia_sub_statislist = []
    divia_sub_averageoddlist = []
    divia_sub_disparate_impactlist = []
    divia_sub_traindifflist = []

    for thisline in lines:
        if "trainsizeratio" in thisline:
            continue
        splits = thisline.split(",")
        feature = splits[1]
        if feature == "1" or feature == "2":
            continue
        traindifflist.append((float(splits[3])))
        metric_list.append((float(splits[4])))
        equaloppfairmetric_list.append((float(splits[5])))
        statislist.append((float(splits[6])))
        averageoddlist.append((float(splits[7])))
        disparate_impactlist.append((float(splits[8])))

        divia_add_traindifflist.append((float(splits[11]) + float(splits[3])))
        divia_add_metric_list.append((float(splits[12]) + float(splits[4])))
        divia_add_equaloppfairmetric_list.append((float(splits[13]) + float(splits[5])))
        divia_add_statislist.append((float(splits[14]) + float(splits[6])))
        divia_add_averageoddlist.append((float(splits[15]) + float(splits[7])))
        divia_add_disparate_impactlist.append((float(splits[16]) + float(splits[8])))

        divia_sub_traindifflist.append(-(float(splits[11]) - float(splits[3])))
        divia_sub_metric_list.append(-(float(splits[12]) - float(splits[4])))
        divia_sub_equaloppfairmetric_list.append(
            -(float(splits[13]) - float(splits[5]))
        )
        divia_sub_statislist.append(-(float(splits[14]) - float(splits[6])))
        divia_sub_averageoddlist.append(-(float(splits[15]) - float(splits[7])))
        divia_sub_disparate_impactlist.append(-(float(splits[16]) - float(splits[8])))

    dataset_orig, privileged_groups, unprivileged_groups = lib.get_data(
        datasetname, protectedattribute
    )
    column_names = dataset_orig.feature_names
    range_ = np.arange(3, len(statislist) + 3, 1)
    print(range_)
    print()
    fig, ax1 = plt.subplots(figsize=(4, 4))
    # import pylab
    # figlegend = pylab.figure(figsize=(13, 1))
    lines = []

    lines += ax1.plot(
        range_, statislist, ".-.", color="b", label="statistical parity", linewidth=3
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
        linewidth=3,
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
        linewidth=3,
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
        linewidth=3,
    )
    ax1.fill_between(
        range_,
        divia_add_disparate_impactlist,
        divia_sub_disparate_impactlist,
        facecolor="green",
        alpha=0.1,
    )

    ax1.set_title(
        datasetname + " - " + protectedattribute.lower(), fontsize=22, fontweight="bold"
    )
    ax1.set_xlabel("", fontsize=22, fontweight="bold")
    ax1.set_ylabel("", color="black", fontsize=22, fontweight="bold")
    ax1.xaxis.set_tick_params(labelsize=22)
    ax1.yaxis.set_tick_params(labelsize=22)

    ax1.set_ylim((-0.1, 1.0))

    from matplotlib.ticker import FormatStrFormatter

    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    if "adult" or "compas" in datasetname:
        plt.xticks(np.arange(min(range_), max(range_) + 1, 2))
    if "bank" or "german" in datasetname:
        plt.xticks(np.arange(min(range_), max(range_) + 1, 4))
    if "meps" in datasetname:
        plt.xticks(np.arange(min(range_), max(range_) + 1, 8))

    plt.savefig(
        "./plots/" + datasetname + "-" + protectedattribute + "-fn-lr.pdf",
        bbox_inches="tight",
    )
    # figlegend.savefig('./plots/' + 'legend.pdf', bbox_inches='tight')
    # figlegend.show()
    # plt.legend()
    plt.show()


def getlist_everymetric(datasetname, protectedattribute, filepath, metricnum):
    newpath = filepath.replace(".csv", "_average.csv")
    readfile = open(newpath)
    lines = readfile.readlines()
    metric_list = []
    divia_add_metric_list = []
    divia_sub_metric_list = []

    for thisline in lines:
        if "trainsizeratio" in thisline:
            continue
        splits = thisline.split(",")
        feature = splits[1]
        if feature == "1" or feature == "0":
            continue

        metric_list.append((float(splits[metricnum])))

        divia_add_metric_list.append(
            (float(splits[metricnum + 8]) + float(splits[metricnum]))
        )
        divia_sub_metric_list.append(
            -(float(splits[metricnum + 8]) - float(splits[metricnum]))
        )

    return [metric_list, divia_add_metric_list, divia_sub_metric_list]


def runall():
    datasetnamelist = [["adult", "race"]]
    for i in datasetnamelist:
        filepath = "./newresults/" + i[0] + "-" + i[1] + "-2d-featurenum-gridsearch.csv"
        collectdata(i[0], i[1], filepath)
        get_average(filepath)
        drawFig(i[0], i[1], filepath)


if __name__ == "__main__":

    runall()
