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
import itertools
from aif360.datasets import AdultDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.explainers import MetricTextExplainer

# Following is an exhaustive list of all possible combinations for
# privileged_groups & unprivileged_groups. The list is of type
# List[Tuple] ie. it contains tuples where each tuple contains 2
# elements. The first element being privileged_groups(p) & the second
# unprivileged_groups(u).

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

# TODO see docs for StandardDataset; it does a number of preprocessing
# steps. We may want to revisit these stratergies.

adult = AdultDataset()

for combination in combinations:
    p, u = combination # unpack tuple

    try:
        metrics = BinaryLabelDatasetMetric(dataset=adult,
                                           unprivileged_groups=u,
                                           privileged_groups=p)
        explainer = MetricTextExplainer(metrics)
        print()
        print()
        print("INFO: printing metrics for p: {}, u: {}".format(p, u))
        print(explainer.num_positives(privileged=None))
        print(explainer.num_positives(privileged=True))
        print(explainer.num_positives(privileged=False))
        print()
        print(explainer.num_negatives(privileged=None))
        print(explainer.num_negatives(privileged=True))
        print(explainer.num_negatives(privileged=False))
        print()
        # MetricTextExplainer does not contain functions for base_rate so
        # I implement them manually here.
        print("Base rate for entire dataset: {}".format(metrics.base_rate(privileged=None)))
        print("Base rate conditioned on privileged group: {}".format(metrics.base_rate(privileged=True)))
        print("Base rate conditioned on unprivileged group: {}".format(metrics.base_rate(privileged=False)))
        print(explainer.disparate_impact())
        print(explainer.statistical_parity_difference())
        # print(explainer.consistency()) # this one takes time
        # print("Smoothed EDF: {}".format(metrics.smoothed_empirical_differential_fairness()))
    except ValueError:
        print("WARNING: p: {}, u: {} are not disjoint".format(p, u))

"""Output of above block:
WARNING:root:Missing Data: 3620 rows removed from AdultDataset.


INFO: printing metrics for p: [{'sex': 1}], u: [{'sex': 0}]
Number of positive-outcome instances: 11208.0
Number of privileged positive-outcome instances: 9539.0
Number of unprivileged positive-outcome instances: 1669.0

Number of negative-outcome instances: 34014.0
Number of privileged negative-outcome instances: 20988.0
Number of unprivileged negative-outcome instances: 13026.0

Base rate for entire dataset: 0.2478439697492371
Base rate conditioned on privileged group: 0.31247747895305794
Base rate conditioned on unprivileged group: 0.11357604627424293
Disparate impact (probability of favorable outcome for unprivileged instances / probability of favorable outcome for privileged instances): 0.3634695423643793
Statistical parity difference (probability of favorable outcome for unprivileged instances - probability of favorable outcome for privileged instances): -0.198901432678815
WARNING: p: [{'sex': 1}], u: [{'race': 0}] are not disjoint


INFO: printing metrics for p: [{'sex': 1}], u: [{'sex': 0, 'race': 0}]
Number of positive-outcome instances: 11208.0
Number of privileged positive-outcome instances: 9539.0
Number of unprivileged positive-outcome instances: 214.0

Number of negative-outcome instances: 34014.0
Number of privileged negative-outcome instances: 20988.0
Number of unprivileged negative-outcome instances: 2598.0

Base rate for entire dataset: 0.2478439697492371
Base rate conditioned on privileged group: 0.31247747895305794
Base rate conditioned on unprivileged group: 0.07610241820768136
Disparate impact (probability of favorable outcome for unprivileged instances / probability of favorable outcome for privileged instances): 0.24354528992828273
Statistical parity difference (probability of favorable outcome for unprivileged instances - probability of favorable outcome for privileged instances): -0.23637506074537656
WARNING: p: [{'sex': 1}], u: [{'sex': 0}, {'race': 0}] are not disjoint
WARNING: p: [{'race': 1}], u: [{'sex': 0}] are not disjoint


INFO: printing metrics for p: [{'race': 1}], u: [{'race': 0}]
Number of positive-outcome instances: 11208.0
Number of privileged positive-outcome instances: 10207.0
Number of unprivileged positive-outcome instances: 1001.0

Number of negative-outcome instances: 34014.0
Number of privileged negative-outcome instances: 28696.0
Number of unprivileged negative-outcome instances: 5318.0

Base rate for entire dataset: 0.2478439697492371
Base rate conditioned on privileged group: 0.2623705112716243
Base rate conditioned on unprivileged group: 0.15841114100332332
Disparate impact (probability of favorable outcome for unprivileged instances / probability of favorable outcome for privileged instances): 0.6037688467181627
Statistical parity difference (probability of favorable outcome for unprivileged instances - probability of favorable outcome for privileged instances): -0.10395937026830099


INFO: printing metrics for p: [{'race': 1}], u: [{'sex': 0, 'race': 0}]
Number of positive-outcome instances: 11208.0
Number of privileged positive-outcome instances: 10207.0
Number of unprivileged positive-outcome instances: 214.0

Number of negative-outcome instances: 34014.0
Number of privileged negative-outcome instances: 28696.0
Number of unprivileged negative-outcome instances: 2598.0

Base rate for entire dataset: 0.2478439697492371
Base rate conditioned on privileged group: 0.2623705112716243
Base rate conditioned on unprivileged group: 0.07610241820768136
Disparate impact (probability of favorable outcome for unprivileged instances / probability of favorable outcome for privileged instances): 0.2900570564841215
Statistical parity difference (probability of favorable outcome for unprivileged instances - probability of favorable outcome for privileged instances): -0.18626809306394293
WARNING: p: [{'race': 1}], u: [{'sex': 0}, {'race': 0}] are not disjoint


INFO: printing metrics for p: [{'sex': 1, 'race': 1}], u: [{'sex': 0}]
Number of positive-outcome instances: 11208.0
Number of privileged positive-outcome instances: 8752.0
Number of unprivileged positive-outcome instances: 1669.0

Number of negative-outcome instances: 34014.0
Number of privileged negative-outcome instances: 18268.0
Number of unprivileged negative-outcome instances: 13026.0

Base rate for entire dataset: 0.2478439697492371
Base rate conditioned on privileged group: 0.3239082161361954
Base rate conditioned on unprivileged group: 0.11357604627424293
Disparate impact (probability of favorable outcome for unprivileged instances / probability of favorable outcome for privileged instances): 0.3506426839956632
Statistical parity difference (probability of favorable outcome for unprivileged instances - probability of favorable outcome for privileged instances): -0.21033216986195247


INFO: printing metrics for p: [{'sex': 1, 'race': 1}], u: [{'race': 0}]
Number of positive-outcome instances: 11208.0
Number of privileged positive-outcome instances: 8752.0
Number of unprivileged positive-outcome instances: 1001.0

Number of negative-outcome instances: 34014.0
Number of privileged negative-outcome instances: 18268.0
Number of unprivileged negative-outcome instances: 5318.0

Base rate for entire dataset: 0.2478439697492371
Base rate conditioned on privileged group: 0.3239082161361954
Base rate conditioned on unprivileged group: 0.15841114100332332
Disparate impact (probability of favorable outcome for unprivileged instances / probability of favorable outcome for privileged instances): 0.48906181785989444
Statistical parity difference (probability of favorable outcome for unprivileged instances - probability of favorable outcome for privileged instances): -0.1654970751328721


INFO: printing metrics for p: [{'sex': 1, 'race': 1}], u: [{'sex': 0, 'race': 0}]
Number of positive-outcome instances: 11208.0
Number of privileged positive-outcome instances: 8752.0
Number of unprivileged positive-outcome instances: 214.0

Number of negative-outcome instances: 34014.0
Number of privileged negative-outcome instances: 18268.0
Number of unprivileged negative-outcome instances: 2598.0

Base rate for entire dataset: 0.2478439697492371
Base rate conditioned on privileged group: 0.3239082161361954
Base rate conditioned on unprivileged group: 0.07610241820768136
Disparate impact (probability of favorable outcome for unprivileged instances / probability of favorable outcome for privileged instances): 0.2349505644391625
Statistical parity difference (probability of favorable outcome for unprivileged instances - probability of favorable outcome for privileged instances): -0.24780579792851404


INFO: printing metrics for p: [{'sex': 1, 'race': 1}], u: [{'sex': 0}, {'race': 0}]
Number of positive-outcome instances: 11208.0
Number of privileged positive-outcome instances: 8752.0
Number of unprivileged positive-outcome instances: 2456.0

Number of negative-outcome instances: 34014.0
Number of privileged negative-outcome instances: 18268.0
Number of unprivileged negative-outcome instances: 15746.0

Base rate for entire dataset: 0.2478439697492371
Base rate conditioned on privileged group: 0.3239082161361954
Base rate conditioned on unprivileged group: 0.13493022744753325
Disparate impact (probability of favorable outcome for unprivileged instances / probability of favorable outcome for privileged instances): 0.4165693265119228
Statistical parity difference (probability of favorable outcome for unprivileged instances - probability of favorable outcome for privileged instances): -0.18897798868866217
WARNING: p: [{'sex': 1}, {'race': 1}], u: [{'sex': 0}] are not disjoint
WARNING: p: [{'sex': 1}, {'race': 1}], u: [{'race': 0}] are not disjoint


INFO: printing metrics for p: [{'sex': 1}, {'race': 1}], u: [{'sex': 0, 'race': 0}]
Number of positive-outcome instances: 11208.0
Number of privileged positive-outcome instances: 10994.0
Number of unprivileged positive-outcome instances: 214.0

Number of negative-outcome instances: 34014.0
Number of privileged negative-outcome instances: 31416.0
Number of unprivileged negative-outcome instances: 2598.0

Base rate for entire dataset: 0.2478439697492371
Base rate conditioned on privileged group: 0.25923131336948835
Base rate conditioned on unprivileged group: 0.07610241820768136
Disparate impact (probability of favorable outcome for unprivileged instances / probability of favorable outcome for privileged instances): 0.29356954304054633
Statistical parity difference (probability of favorable outcome for unprivileged instances - probability of favorable outcome for privileged instances): -0.18312889516180697
WARNING: p: [{'sex': 1}, {'race': 1}], u: [{'sex': 0}, {'race': 0}] are not disjoint
"""

"""Above output in simpler terms:
p: [{'sex': 1}], u: [{'sex': 0}]                           ARE     disjoint
p: [{'sex': 1}], u: [{'sex': 0, 'race': 0}]                ARE     disjoint
p: [{'race': 1}], u: [{'race': 0}]                         ARE     disjoint
p: [{'race': 1}], u: [{'sex': 0, 'race': 0}]               ARE     disjoint
p: [{'sex': 1, 'race': 1}], u: [{'sex': 0}]                ARE     disjoint
p: [{'sex': 1, 'race': 1}], u: [{'race': 0}]               ARE     disjoint
p: [{'sex': 1, 'race': 1}], u: [{'sex': 0, 'race': 0}]     ARE     disjoint
p: [{'sex': 1, 'race': 1}], u: [{'sex': 0}, {'race': 0}]   ARE     disjoint
p: [{'sex': 1}, {'race': 1}], u: [{'sex': 0, 'race': 0}]   ARE     disjoint

p: [{'sex': 1}], u: [{'race': 0}]                          ARE NOT disjoint
p: [{'sex': 1}], u: [{'sex': 0}, {'race': 0}]              ARE NOT disjoint
p: [{'race': 1}], u: [{'sex': 0}]                          ARE NOT disjoint
p: [{'race': 1}], u: [{'sex': 0}, {'race': 0}]             ARE NOT disjoint
p: [{'sex': 1}, {'race': 1}], u: [{'sex': 0}]              ARE NOT disjoint
p: [{'sex': 1}, {'race': 1}], u: [{'race': 0}]             ARE NOT disjoint
p: [{'sex': 1}, {'race': 1}], u: [{'sex': 0}, {'race': 0}] ARE NOT disjoint
"""
