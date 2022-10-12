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
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import lib

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

    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))

    return metrics


def collectdata(datasetname, protectedattribute, turnrange, filepath):
    dataset_orig, privileged_groups, unprivileged_groups = lib.get_data(
        datasetname, protectedattribute
    )

    originalfeatureset = dataset_orig.feature_names
    featureset = originalfeatureset[:]
    featuresubset_init = []
    #
    writefile = open(filepath, "w")
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
    writefile.close()

    protected_attribute_names = [protectedattribute]
    protected_attribute_index = -1
    for each in protected_attribute_names:
        protected_attribute_index = featureset.index(each)
        featuresubset_init.append(featureset.index(each))
    for each in protected_attribute_names:
        featureset.remove(each)

    featurenumlist = np.arange(2, 11, 1)

    depthlist = [10]

    trainingdatasizelist = np.arange(0.1, 1, 0.1)

    for turn in turnrange:
        seedr = random.randint(0, 100)
        dataset_orig_train_total, dataset_orig_test = dataset_orig.split(
            [0.8], shuffle=True, seed=seedr
        )
        print("======================================TURN: " + str(turn))

        for trainsizeratio in trainingdatasizelist:
            print("training data size: " + str(trainsizeratio))
            seedr = random.randint(0, 100)
            dataset_orig_train, dataset_orig_mmm = dataset_orig_train_total.split(
                [trainsizeratio], shuffle=True, seed=seedr
            )

            dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
            dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

            for numfeatures in featurenumlist:
                print("num of features: " + str(numfeatures))
                featuresubset = list(np.copy(featuresubset_init))

                for feature in featureset[:numfeatures]:
                    featuresubset.append(originalfeatureset.index(feature))

                scale_orig = StandardScaler()
                featuresubset = list(set(featuresubset))

                X_train_fullfeature = scale_orig.fit_transform(
                    dataset_orig_train.features
                )
                y_train = dataset_orig_train.labels.ravel()
                featurecolumn = dataset_orig_train.features[
                    :, protected_attribute_index
                ]
                X_train = X_train_fullfeature[:, featuresubset]

                metric_orig_train = BinaryLabelDatasetMetric(
                    dataset_orig_train,
                    unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups,
                )

                for depth in depthlist:

                    lmod = tree.DecisionTreeClassifier(max_depth=depth)
                    lmod.fit(X_train, y_train)

                    pv = lib.get_PV_classic(lmod, X_train, y_train)

                    fav_idx = np.where(
                        lmod.classes_ == dataset_orig_train.favorable_label
                    )[0][0]
                    y_train_pred_prob = lmod.predict_proba(X_train)[:, fav_idx]

                    X_test_fullfeature = scale_orig.transform(
                        dataset_orig_test.features
                    )
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

                    writefile = open(filepath, "a")

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
                        + ","  # 5
                        + str(cm_transf_test.accuracy())
                        + ","
                        + str(cm_transf_test.equal_opportunity_difference())
                        + ","
                        + str(cm_transf_test.statistical_parity_difference())
                        + ","  # 8
                        + str(cm_transf_test.average_abs_odds_difference())
                        + ","
                        + str(cm_transf_test.disparate_impact())
                        + "\n"
                    )
                    writefile.close()


def get_average(filepath, metricstring):
    readfile = open(filepath)
    lines = readfile.readlines()
    newfilepath = filepath.replace(".csv", "_average.csv")
    writefile = open(newfilepath, "w")
    writefile.write(
        "trainsizeratio,featurenum,depth,equalopp,trainmeandiff,testaccuracy\n"
    )
    dic_string_valuestring = {}
    dic_string_meandiff = {}
    dic_string_testaccu = {}
    for thisline in lines:

        if "datasetname" in thisline:
            continue
        splits = thisline.split(",")
        turn = float(splits[3])
        if turn < 3:
            continue
        threestring = splits[2] + "," + splits[3] + "," + splits[4]
        if threestring in dic_string_valuestring:
            if "equal" in metricstring:
                dic_string_valuestring[threestring].append(abs(float(splits[7])))
            if "statistical" in metricstring:
                dic_string_valuestring[threestring].append(abs(float(splits[8])))
            if "absolute" in metricstring:
                dic_string_valuestring[threestring].append(abs(float(splits[9])))
            if "disparate" in metricstring:
                dic_string_valuestring[threestring].append(abs(1 - float(splits[10])))
            if "test accuracy" in metricstring:
                dic_string_valuestring[threestring].append(abs(float(splits[6])))
            dic_string_meandiff[threestring].append((float(splits[5])))
            dic_string_testaccu[threestring].append((float(splits[9])))
        else:
            dic_string_valuestring[threestring] = [(float(splits[10]))]
            dic_string_meandiff[threestring] = [(float(splits[5]))]
            dic_string_testaccu[threestring] = [(float(splits[9]))]

    for each in dic_string_valuestring:
        thislist = dic_string_valuestring[each]
        thatlist = dic_string_meandiff[each]
        testacclist = dic_string_testaccu[each]
        writefile.write(
            each
            + ","
            + str(1.0 * sum(thislist) / len(thislist))
            + ","
            + str(1.0 * sum(thatlist) / len(thatlist))
            + ","
            + str(1.0 * sum(testacclist) / len(testacclist))
            + "\n"
        )
    writefile.close()


def drawSlide(datasetname, protectedattribute, string, filepath, metricstring):
    df_gridsearch = pandas.read_csv(filepath)
    min_samples_split_list = list(set(df_gridsearch.depth))
    min_samples_split_list.sort()

    data = []
    for n in min_samples_split_list:
        filtered_df = df_gridsearch[df_gridsearch.depth == n]
        scores = filtered_df.pivot(
            "trainsizeratio",
            "featurenum",
        )
        traces = [
            go.Surface(
                z=scores[set_name].values,
                x=scores[set_name].columns,
                y=scores.index,
                colorscale="Viridis",
                showscale=False,
                hoverinfo="text+name",
                # text=scores.Text.values,
                name=set_name,
                visible=False,
            )
            for set_name in [string, string]
        ]

        data.append(traces[0])
        data.append(traces[1])

    data[0].visible = True
    data[1].visible = True

    steps = []
    for i, n in enumerate(min_samples_split_list):
        step = dict(method="restyle", args=["visible", [False] * len(data) * 2])
        step["args"][1][i * 2] = True  # toggle i'th traces to 'visible'
        step["args"][1][i * 2 + 1] = True  # toggle i'th traces to 'visible'
        step["label"] = str(n)
        steps.append(step)
    sliders = [
        dict(
            active=4,
            currentvalue={"prefix": "depth: "},
            pad={"t": 10, "b": 20},
            steps=steps,
            len=0.5,
            xanchor="center",
            x=0.5,
        )
    ]
    layout = go.Layout(
        # title=string+' VS maximum_depth and num_feature',
        margin=dict(l=100, r=100, b=100, t=100),
        # height=100,
        # width=260,
        sliders=sliders,
        scene=dict(
            xaxis=dict(title="feature set size", nticks=5),
            yaxis=dict(title="training data size", nticks=5),
            zaxis=dict(
                title=metricstring,
                # range=[0.1, 0.3],
                nticks=5,
            ),
            #         aspectratio = dict(
            #             x = 1.3,
            #             y = 1.3,
            #             z = .9
            #         ),
            aspectratio=dict(
                x=0.5,
                y=0.5,
                z=0.5,
            ),
            camera=dict(
                eye=dict(
                    y=-1.089757339892154,
                    x=-0.8464711077183096,
                    z=0.54759264478960377,
                )
            ),
        ),
    )

    fig = go.Figure(data=data, layout=layout)
    fig.write_image(
        "./../plots/"
        + datasetname
        + "-"
        + protectedattribute
        + "-3d-"
        + metricstring.split(" ")[0]
        + ".pdf"
    )

    plot(fig)


def runall():
    datasetnamelist = [["adult", "sex"], ["compas", "race"]]
    metricstring = ["test accuracy"]
    for i in datasetnamelist:
        filepath = "./../results/" + i[0] + "-" + i[1] + "-3d-mutirun.csv"
        for j in metricstring:
            get_average(filepath, j)
            drawSlide(
                i[0], i[1], "equalopp", filepath.replace(".csv", "_average.csv"), j
            )


if __name__ == "__main__":
    runall()
