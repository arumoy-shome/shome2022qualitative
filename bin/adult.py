"""Experiments with both protected attribute sex & race in adult dataset.

This file contains experiments for both sex & race protected
attribute. We consider all permutations & combinations for the
privileged_groups & unprivileged_groups.

The BinaryLabelDatasetMetric expects the {un,}privileged_groups
parameters to be list of dicts. Where the dicts contain as keys the
name of the protected attribute
(AdultDataset.protected_attribute_names) & as value the values of the
protected attribute (AdultDataset.protected_attributes).

"""

from aif360.datasets import AdultDataset
from aif360.explainers import MetricTextExplainer
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import csv
import itertools
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

# TODO see docs for StandardDataset; it does a number of preprocessing
# steps. We may want to revisit these stratergies.

adult = AdultDataset()
train, test = adult.split([0.75], shuffle=True, seed=42)
pipe = make_pipeline(StandardScaler(),
                     LogisticRegression())
pipe.fit(X=train.features,
         y=train.labels.ravel())
y_truth = test.labels.ravel()
y_pred = pipe.predict(test.features).reshape(-1, 1) # ClassificationMetric expects 1D column vector
classified = test.copy() # as shown in this tutorial
                         # <https://nbviewer.org/github/IBM/AIF360/blob/master/examples/tutorial_medical_expenditure.ipynb>
classified.labels = y_pred

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

header = ['dataset', 'subset', 'model', 'privileged',
          'privileged_groups', 'unprivileged_groups', 'num_positives',
          'num_negatives', 'base_rate', 'disparate_impact',
          'statistical_parity_difference', 'consistency',
          'smoothed_empirical_differential_fairness',
          'num_true_positives', 'num_false_positives',
          'num_false_negatives', 'num_true_negatives',
          'num_generalized_true_positives',
          'num_generalized_false_positives',
          'num_generalized_false_negatives',
          'num_generalized_true_negatives', 'true_positive_rate',
          'false_positive_rate', 'false_negative_rate',
          'true_negative_rate', 'generalized_true_positive_rate',
          'generalized_false_positive_rate',
          'generalized_false_negative_rate',
          'generalized_true_negative_rate',
          'positive_predictive_value', 'false_discovery_rate',
          'false_omission_rate', 'negative_predictive_value',
          'accuracy', 'error_rate', 'true_positive_rate_difference',
          'false_positive_rate_difference',
          'false_negative_rate_difference',
          'false_omission_rate_difference',
          'false_discovery_rate_difference',
          'false_positive_rate_ratio', 'false_negative_rate_ratio',
          'false_omission_rate_ratio', 'false_discovery_rate_ratio',
          'average_odds_difference', 'average_abs_odds_difference',
          'error_rate_difference', 'error_rate_ratio',
          'num_pred_positives', 'num_pred_negatives',
          'selection_rate', 'generalized_entropy_index',
          'between_all_groups_generalized_entropy_index',
          'between_group_generalized_entropy_index', 'theil_index',
          'coefficient_of_variation', 'between_group_theil_index',
          'between_group_coefficient_of_variation',
          'between_all_groups_theil_index',
          'between_all_groups_coefficient_of_variation',
          'differential_fairness_bias_amplification']

with open('adult.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()

    p_permutations = [
        [{'sex': 1}],
        [{'race': 1}],
        [{'sex': 1, 'race': 1}],      # sex == 1 AND race == 1
        [{'sex': 1}, {'race': 1}]     # sex == 1 OR  race == 1
    ]

    u_permutations = [
        [{'sex': 0}],
        [{'race': 0}],
        [{'sex': 0, 'race': 0}],      # sex == 0 AND race == 0
        [{'sex': 0}, {'race': 0}]     # sex == 0 OR race == 0
    ]

    combinations = itertools.product(p_permutations, u_permutations)

    datasets = [('full', adult), ('train', train), ('test', test)]
    conditions = [True, False, None]

    for combination in combinations:
        p, u = combination

        logging.info("**********DATASET METRICS**********")
        for label, dataset in datasets:
            try:
                metrics = BinaryLabelDatasetMetric(dataset=dataset,
                                                   privileged_groups=p,
                                                   unprivileged_groups=u)

                for condition in conditions:
                    logging.info("p: {}, u: {}, dataset: {}, condition: {}".format(p, u, label, condition))
                    row = {}
                    for col in header: row[col] = np.nan # fill everything with np.nan
                    row['dataset'] = 'adult'
                    row['subset'] = label
                    row['privileged'] = condition
                    row['privileged_groups'] = p
                    row['unprivileged_groups'] = u

                    row['num_positives'] = metrics.num_positives(privileged=condition)
                    row['num_negatives'] = metrics.num_negatives(privileged=condition)
                    row['base_rate'] = metrics.base_rate(privileged=condition)
                    if condition is None:
                        row['disparate_impact'] = metrics.disparate_impact()
                        row['statistical_parity_difference'] = metrics.statistical_parity_difference()
                        row['consistency'] = metrics.consistency()
                        row['smoothed_empirical_differential_fairness'] = metrics.smoothed_empirical_differential_fairness()

                    writer.writerow(row)

            except ValueError:
                logging.warning("p: {}, u: {} ARE NOT disjoint".format(p, u))

        logging.info("**********MODEL METRICS**********")
        try:
            metrics = ClassificationMetric(dataset=test,
                                           classified_dataset=classified,
                                           privileged_groups=p,
                                           unprivileged_groups=u)

            for condition in conditions:
                logging.info("p: {}, u: {}, dataset: {}, condition: {}".format(p, u, label, condition))
                row = {}
                for col in header: row[col] = np.nan
                row['dataset'] = 'adult'
                row['subset'] = 'test'
                row['privileged'] = condition
                row['privileged_groups'] = p
                row['unprivileged_groups'] = u
                row['model'] = 'lr' # REFACTOR use name from pipe instead

                row['num_true_positives'] = metrics.num_true_positives(privileged=condition)
                row['num_false_positives'] = metrics.num_false_positives(privileged=condition)
                row['num_false_negatives'] = metrics.num_false_negatives(privileged=condition)
                row['num_true_negatives'] = metrics.num_true_negatives(privileged=condition)
                row['num_generalized_true_positives'] = metrics.num_generalized_true_positives(privileged=condition)
                row['num_generalized_false_positives'] = metrics.num_generalized_false_positives(privileged=condition)
                row['num_generalized_false_negatives'] = metrics.num_generalized_false_negatives(privileged=condition)
                row['num_generalized_true_negatives'] = metrics.num_generalized_true_negatives(privileged=condition)
                row['true_positive_rate'] = metrics.true_positive_rate(privileged=condition)
                row['false_positive_rate'] = metrics.false_positive_rate(privileged=condition)
                row['false_negative_rate'] = metrics.false_negative_rate(privileged=condition)
                row['true_negative_rate'] = metrics.true_negative_rate(privileged=condition)
                row['generalized_true_positive_rate'] = metrics.generalized_true_positive_rate(privileged=condition)
                row['generalized_false_positive_rate'] = metrics.generalized_false_positive_rate(privileged=condition)
                row['generalized_false_negative_rate'] = metrics.generalized_false_negative_rate(privileged=condition)
                row['generalized_true_negative_rate'] = metrics.generalized_true_negative_rate(privileged=condition)
                row['positive_predictive_value'] = metrics.positive_predictive_value(privileged=condition)
                row['false_discovery_rate'] = metrics.false_discovery_rate(privileged=condition)
                row['false_omission_rate'] = metrics.false_omission_rate(privileged=condition)
                row['negative_predictive_value'] = metrics.negative_predictive_value(privileged=condition)
                row['accuracy'] = metrics.accuracy(privileged=condition)
                row['error_rate'] = metrics.error_rate(privileged=condition)
                row['num_pred_positives'] = metrics.num_pred_positives(privileged=condition)
                row['num_pred_negatives'] = metrics.num_pred_negatives(privileged=condition)
                row['selection_rate'] = metrics.selection_rate(privileged=condition)

                if condition is None:
                    row['true_positive_rate_difference'] = metrics.true_positive_rate_difference()
                    row['false_positive_rate_difference'] = metrics.false_positive_rate_difference()
                    row['false_negative_rate_difference'] = metrics.false_negative_rate_difference()
                    row['false_omission_rate_difference'] = metrics.false_omission_rate_difference()
                    row['false_discovery_rate_difference'] = metrics.false_discovery_rate_difference()
                    row['average_odds_difference'] = metrics.average_odds_difference()
                    row['average_abs_odds_difference'] = metrics.average_abs_odds_difference()
                    row['error_rate_difference'] = metrics.error_rate_difference()
                    row['error_rate_ratio'] = metrics.error_rate_ratio()
                    row['disparate_impact'] = metrics.disparate_impact()
                    row['statistical_parity_difference'] = metrics.statistical_parity_difference()
                    row['generalized_entropy_index'] = metrics.generalized_entropy_index()
                    row['between_all_groups_generalized_entropy_index'] = metrics.between_all_groups_generalized_entropy_index()
                    row['between_group_generalized_entropy_index'] = metrics.between_group_generalized_entropy_index()
                    row['theil_index'] = metrics.theil_index()
                    row['coefficient_of_variation'] = metrics.coefficient_of_variation()
                    row['between_group_theil_index'] = metrics.between_group_theil_index()
                    row['between_group_coefficient_of_variation'] = metrics.between_group_coefficient_of_variation()
                    row['between_all_groups_theil_index'] = metrics.between_all_groups_theil_index()
                    row['between_all_groups_coefficient_of_variation'] = metrics.between_all_groups_coefficient_of_variation()
                    row['differential_fairness_bias_amplification'] = metrics.differential_fairness_bias_amplification()
                writer.writerow(row)

        except ValueError:
            logging.warning("p: {}, u: {} ARE NOT disjoint".format(p, u))
