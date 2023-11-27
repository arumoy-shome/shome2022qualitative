Dear authors,

The IEEE SaTML committee is sorry to inform you that your paper #35 has been early rejected and will not appear in the conference. The selection process was competitive, even in the first review round. We hope that your submission will find a good home and that you will be able to present your results to the community at another venue.

Reviews for your paper are copied at the end of this email. We hope they will be useful to you in revising your work. We thank you for submitting your work to IEEE SaTML and hope that you will consider doing so again in the future.

Best,

Carmela and Nicolas.

# Review 1:

## Summary
This manuscript empirically investigates the correlation between model-free fairness evaluation (using the dataset only) and model-based fairness evaluation. The primary motivation is to consider:

What relationships exist for fairness metrics that can be computed using training data alone vs. modeled fairness metrics?
The effects of training data size on fairness metrics.
The relationship between approaches for fairness metric evaluation with various training and feature sizes.
The evaluation focused on a disparate impact metric (ratio of the group-conditional probability of positive predictions) and a statistical parity metric (difference between the group-conditional probability of positive predictions).
Overall, the claim that the suggested metrics might be closely related is not surprising, and the metrics of interest depend only on predictions and should get closer as the data and prediction get closer (actually, a weaker sufficient requirement is that the probability of group-conditional positive prediction should be close to the training data). From standard generalization arguments, one expects this to be truer as the model improves (more training data, better generalization). However, some of the main results conflict with this intuition, and it is unclear why.

In my opinion, the evaluation is interesting, but it is less clear that the results offer much insight or guidance to practice. I also think some of the main results may have errors.

## Strengths:

the key idea that one might predict fairness properties directly from training data is an interesting one in the abstract, as it is attractive to avoid model fitting as a confound. One imagines this approach might be more efficient in some instances.
The evaluation seems thorough for the datasets considered. Unfortunately, there are few fairness datasets available, but the authors have covered most of the public ones
Minor points:

Arguably, the model-free method is itself a choice of model, i.e., quite close to the nearest neighbor predictor. This perspective may also be helpful for further evaluation.
the authors mentioned the "fairness evaluation cycles," i.e., the number of evaluations -- several times -- I speculate that this highlighted the amount of work this must have taken. While I appreciate the effort here, the amount of computation alone is generally not a metric for manuscript quality.
Questions that would be helpful to answer in a resubmission of your work:

## Weaknesses:

the main result that DFM (fairness from the data) and MFM (fairness from the model) diverge with large training samples seems incorrect, as one expects the measures to be closer as the training data increases. Hence, the estimates of group conditional positive probabilities are closer. The authors are encouraged to further ablate, check, or justify this result. I am especially interested in a plot of accuracy vs fairness (again, more accurate models should more closely reflect the data, so the gap should narrow with more data).
In some ways, the observed correlation seems more obvious for fairness metrics that depend on group conditional positive probabilities; the more interesting question is if there is a mechanism for computing the same correlation if the metrics are more complicated, e.g., equal opportunity, which involves both the true labels and the prediction. There, the mechanisms for correlation are less clear.
I worry about a false sense of confidence if one builds systems following this analysis -- as there are model fitting choices that can make the prediction fairness much worse than the model fairness, even using the metrics chosen.
Rating: 1: Reject

# Review 2:

The authors propose a metric to assess the fairness of a dataset before the model training and evaluate how this metric correlates with the fairness evaluation from model predictions. The paper is well written, and the authors have investigated the effect of the training sample size and feature size on the relationship between model and data fairness. I overall found the paper lacked a clear understanding of fairness metrics, which is exemplified by a low relevant scholarship, despite the extensive Related Works and references. I provide examples below to highlight what I mean, but this is not an exhaustive list.

## Correctness:

"Existing fairness metrics are restricted to supervised binary classification problems". This is simply not true. Multiple methods exist for multiclass problems, regression tasks, ranking systems, ...
'There is a growing consensus amongst academics that not all fairness metrics can be satisfied simultaneously.' Multiple studies now question this statement. (e.g. 2).
In addition, the impact of the metric proposed is unclear:

The metrics selected by the authors are of the 'independence' type (3), because only these can be estimated from the data prior to training. However, these would ignore other biases such as separation or sufficiency.
Models can 'ignore' biases in the data (1). Preventing training such models because of potential biases does not seem to be a solution, although it is always useful to raise awareness to potential issues downstream training.
There is little discussion of this in the text.

## Significance:
The authors do not mention all of the data exploration tools that exist for investigating biases in the data. Tools such as 'Know Your Data' have been deployed to assess biases in data prior to modelling, along with other tools like Data Sheets. Compared to those tools and the various indicators they provide, I do not see a benefit from adopting the metric proposed.

(1) Glocker, B., Jones, C., Bernhardt, M. & Winzeck, S. Algorithmic encoding of protected characteristics in chest X-ray disease detection models. eBioMedicine89, 104467 (2023).
(2) Sanghamitra Dutta, Dennis Wei, Hazar Yueksel, Pin-Yu Chen, Sijia Liu, Kush Varshney Proceedings of the 37th International Conference on Machine Learning, PMLR 119:2803-2813, 2020.
(3) FAIRNESS AND MACHINE LEARNING, Limitations and Opportunities. Solon Barocas, Moritz Hardt, Arvind Narayanan

## Questions that would be helpful to answer in a resubmission of your work:

I cannot recommend this paper for acceptance, even after major revision given the lack of scholarship/understanding in this domain and the low contribution of their proposal. I appreciate the efforts the authors put in their comparisons and further investigations, and would recommend a read of (3) before designing their next research question in the space of fairness.

Rating: 1: Reject

# Review 3:

## Summary:
This paper empirically investigates the effectiveness of pre-training versus post-training de-biasing frameworks. They find that pre-training interventions are effective and are likely to be lower-cost than their post-training counterparts.

## Pros:

## The study provides actionable insights for early detection of biases, which is valuable for ML development.

## Cons:

while the takeaways from this work are interesting and could be helpful in determining the most cost-effective manners of intervening for meaningful fairness change, these takeaways are based on examples only across 5 datasets. in a purely empirical paper, even more datasets would be helpful.
additionally, while I appreciate the steps the authors took to make the findings robust--50 runs for each experiment, and measuring p values-- it would be helpful to see more measures of the reliability of the results across the different datasets.
additionally, it would be helpful to take advantage of the variance of attributes across the different datasets to shed some light on how certain common data bias problems---from underrepresentativeness, to different amounts of noise in features for certain groups, etc, impact the results of the study.
Questions that would be helpful to answer in a resubmission of your work:

N/A

Rating: 1: Reject
