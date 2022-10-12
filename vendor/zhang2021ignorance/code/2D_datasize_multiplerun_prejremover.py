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
from aif360.algorithms.inprocessing import PrejudiceRemover
import json
from aif360.explainers import MetricTextExplainer, MetricJSONExplainer
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import lib
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.metrics.utils import compute_boolean_conditioning_vector
import matplotlib.pyplot as plt
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import (
    load_preproc_data_adult,
    load_preproc_data_compas,
)
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import (
    get_distortion_adult,
)
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression

from IPython.display import Markdown, display
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

    for turn in np.arange(0, 50, 1):
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
            # Placeholder for predicted and transformed datasets

            featuresubset = list(np.copy(featuresubset_init))
            numfeatures = len(dataset_orig.feature_names)
            # numfeatures =3

            # Logistic regression classifier and predictions for training data
            scale_orig = StandardScaler()

            X_train_fullfeature = scale_orig.fit_transform(dataset_orig_train.features)

            dataset_orig_train.features = scale_orig.fit_transform(
                dataset_orig_train.features
            )

            metric_orig_train = BinaryLabelDatasetMetric(
                dataset_orig_train,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
            )

            testdataset = dataset_orig_test.copy()
            testdataset.features = scale_orig.transform(testdataset.features)

            for depth in depthlist:
                # print('depth: '+str(depth))

                lmod = PrejudiceRemover(sensitive_attr=protectedattribute, eta=25.0)
                # lmod = LogisticRegression()
                models = lmod.fit(dataset_orig_train)
                y_val_pred_prob = models.predict(testdataset).scores
                y_val_pred = (y_val_pred_prob[:, 0] > 0.5).astype(np.float64)
                dataset_pred = testdataset.copy()
                dataset_pred.labels = y_val_pred

                cm_transf_test = ClassificationMetric(
                    testdataset,
                    dataset_pred,
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
    datasetnamelist = [["meps", "RACE"]]
    for i in datasetnamelist:
        filepath = "./newresults/" + i[0] + "-" + i[1] + "-2d-datasize-prejremover.csv"
        collectdata(i[0], i[1], filepath)
        lib.get_average(filepath)
        lib.drawFig(i[0], i[1], filepath)


if __name__ == "__main__":

    runall()
