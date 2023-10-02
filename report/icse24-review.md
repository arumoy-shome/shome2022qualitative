---
title: ICSE 2024 reviewer feedback
---
Dear Arumoy Shome,

Thank you for your submission to ICSE2024 early submissions. We are
writing to you about your submission:

,* Title: Data vs. Model Machine Learning Fairness Testing: An
Empirical  Study ,* Site:
https://urldefense.com/v3/__https://icse2024early.hotcrp.com/paper/509__;!!PAKc-5URQlI!7ASWNO9D4HobJNbCgtZ3gSZHaRo6xakF1OLoceg-KolDNG2dNGTvjRV_8zNXRJORkgdN66zH3m_PINb7XUIMZJedXDXL$
* Authors: Arumoy Shome (Delft University of Technology); Luís Cruz
(Delft University of Technology); Arie van Deursen (Delft University
of Technology)

The ICSE2024 Program Committee (PC) has decided  that your paper will
not be accepted and will not undergo revision for ICSE2024. Thus, it
will not be considered further for ICSE2024.

After a rigorous review and discussion process (which, for each paper,
involved at least three program committee members and at least one
area chair), 6.6% of 274 submissions were accepted or conditionally
accepted, 19.7% received a revision decision, while 73.7% were
rejected.  Revisions are an option for the first time in ICSE2024.

We hope you find the reviews and meta-review useful to advance your
work in the future.

Kindly note that your paper cannot be immediately submitted to the
next submission cycle of ICSE 2024, as per the Call for Papers
https://urldefense.com/v3/__https://conf.researchr.org/track/icse-2024/icse-2024-research-track__;!!PAKc-5URQlI!7ASWNO9D4HobJNbCgtZ3gSZHaRo6xakF1OLoceg-KolDNG2dNGTvjRV_8zNXRJORkgdN66zH3m_PINb7XUIMZP8bopdu$
Please take note of the Call for Papers, specifically the paragraph
under “Re-submissions of rejected papers” to get access to the
detailed guidelines.

We request you to consider submissions to other tracks at ICSE 2024
(those with a forthcoming submission date), workshops, and co-located
events. Please check the web page for the submission dates for other
tracks:
https://urldefense.com/v3/__https://conf.researchr.org/home/icse-2024__;!!PAKc-5URQlI!7ASWNO9D4HobJNbCgtZ3gSZHaRo6xakF1OLoceg-KolDNG2dNGTvjRV_8zNXRJORkgdN66zH3m_PINb7XUIMZCXFt3ve$
Our sincere thanks for your kind interest in ICSE2024. We hope you
will still be able to attend the conference in Portugal and we really
hope to see you there !


Best regards
- Abhik Roychoudhury and Margaret Anne-Storey
ICSE 2024 Program Co-Chairs

# Review 509A

Overall merit
1. Reject

## Paper summary
In a nutshell, the paper compares demographic parity (statistical
independence) of data sampled from a dataset with that produced by
a model trained and evaluated in the same data distribution. It finds
that the measured results are fairly similar and recommends that
fairness could be evaluated on the data rather than the model to save
computational effort.

## Overall Comments for authors
Fairness measurement has received a lot of attention in the machine
learning community and to some degree also in software engineering. It
is an important part of considering fairness and ethics in software
systems. This paper encourages to focus on fairness measurement in
data, which is possibly a fresh perspective in the software
engineering literature.

At the same time, the paper’s approach to fairness is shallow and
narrow. The findings are obvious by construction. Finally it is not
clear that the paper actually makes any contribution to software
engineering (as opposed to data science).

## Main point: Of course there is a correlation
The paper focuses on a single fairness metric, here called group
fairness, otherwise also known as demographic parity and statistically
as independence. It simply evaluates whether two groups achieve
similar rates of positive outcomes (the paper uses multiple variants
of this metric). Note that this metric does not care about accuracy of
the predictions of whether those outcomes are justified or fair.

The experiments are set up so that all data is drawn from the same
dataset/distribution, which main or may not contain bias according to
this notion of fairness. The fairness on data is evaluated on the same
distribution from which the model is learned, and the model is
evaluated on predictions for data from the same distribution again. If
we were able to train a perfectly accurate model in that distribution
we would expect, **by construction**, that the outcome disparities are
exactly the same between the original data and the predictions. The
paper does not perform any debiasing steps that would lead us to
expect any other results. Since the trained models are never 100%
accurate and rates are compared on different samples from the same
distributions, the paper not surprisingly finds a strong but not
perfect correlation. All experiments in the paper seem to boil down to
variations of the same theme where we use different parts of the data
for model training and hence vary model accuracy. None of this has
anything to do with fairness. None of this has anything to do with
software engineering. This is all about variance in different sampling
strategies and ability of different models to learn accurate
predictions on different subsets of data. I suspect much of this could
be formally discussed with statistics too.

## Minor comments
* The introduction has lots of questionable claims.
* Fairness and interpretabiliy have been ignored? Really, when there
are entire conferences dedicated to these topics and thousands of
papers published yearly? How are the citations supporting this claim,
e.g., [56]?
* “all existing solutions evaluate fairness after the training stage”
there are lots of discussions of fairness in datasets, e.g., look at
things like Datasheets for Datasets and the hundreds of debiasing and
outlier detection papers that work on data
* “Testing ML systems is also expensive since it involves a full
training-testing cycle” the form of testing discussed here is simply
observing rates of predictions on test or production data -- that
doesn’t seem particularly expensive.
* Biased data does not imply biased models and biased models do not
imply biased products. There are interventions that can be applied at
every of these stages. Most debiasing approaches actually affect the
learning or inference stage to overcome known biases in the training
data. Interventions at the system level are commonly discussed, such
as suppressing entire output categories or adjusting thresholds.
* “Our results are exploratory” what does it mean for results (rather
than methods) to be exploratory?  * What would you consider as
“correctness” of a model? Accuracy?
* Consider describing the intended contributions in the introduction.
* The related work section highlights a narrow and often flawed
understanding of the current fairness discourse
* What is considered as the fairness of a label here? Who decides what
label is fair?
* Fairness metrics are very much not restricted to binary
classification problems
* “ An ML model is said to make unfair decisions if it favours
a certain group or individual pertaining to one or more protected
attributes in the dataset.” -- what does “favours” mean here? This
statement is almost certainly wrong in its generality.
* The exclusive focus on “group fairness” is justified by popularity
and ease of computation of this metric? That’s a very weak
argument. The much more important reason seems to be that this entire
paper would not be possible with other metrics that rely on any notion
of accuracy.
* Note that the distinction into individual fairness and group
fairness with those particular definitions is a rather unusual outlier
commonly repeated in software engineering literature on fairness
that’s almost entirely absent anywhere else.  * “There is a growing
consensus amongst academics that not all fairness metrics can be
satisfied simultaneously” -- this is a mathematical fact that has been
proven (and is fairly obvious)
* “There is also a consensus that fairness and performance of ML
systems are orthogonal to one another and involve a trade-off.” --
this in contrast is quite disputed. Some argue that the tradeoff only
occurs due to a wrong notion of “performance”.
* The entire related work section does not clearly map citations to
claims. Citations seem to be mostly added at the end of a paragraph as
a cluster.
* The discussion of fairness tools seems largely pointless without
mentioning the corresponding fairness properties/metrics.
* “in the recent shift towards application of ML to safety critical
domains, the ground truth must be established through data collection,
cleaning and labelling [4, 13, 43].” -- it is entirely unclear to this
reviewer how the cited papers relate to this claim.
* “With the data ready, a highly experimental phase begins.” -- many
studies show that real data science (outside of student homework
assignments) almost never follows this waterfall-like process.
* “We include all group fairness metrics” -- It is unclear what is the
baseline for the word “all” here. There are many different ways to
measure what’s loosely described as group fairness. Does this just
mean all that are readily implemented in AIF360?
* Why the limit to a single protected attribute at a time?
* “We adopted the transformation steps from prior work“ -- what steps
are those?
* “We extend the above experiment further in two ways.” -- The paper
focuses heavily on the “what” but rarely discusses the “why”. Why this
extension?
* “A positive correlation means that the DFM and MFM changed in the
same direction and thus convey the same information” -- that’s really
not how to correlations work
* Section 4 intermixes methods and results under the label
“Results”. Consider explicitly separating experiment design from
results.
* “value” is not a great axis label in Fig 5
* Many Figures, e.g. Fig 6, are way too small to read
* “Our results indicate that DFM can be used as a early warning system
to identify fairness related data drifts in automated ML pipelines.”
-- unclear how this would work, given that labels would be needed for
all data
* While training can be expensive for some modern ML tasks, training
cost seems to be a very weak argument for the experiments conducted in
this paper. This entire argument also falls apart quickly if any
debiasing steps are taken in learning or when trying to evaluate the
model in production (i.e., moving beyond i.i.d. evaluations).
* “However, we did not find any Python library for fairness testing
which provides a data-centric implementation of these metrics.“ -- why
would there be? This is basic exploratory data analysis. * The Threats
to Validity section include mitigation strategies that are not
discussed in the experiment design earlier. If you mitigated issues
they are no longer threats; discuss mitigations as part of the design
and leave the Threats to Validity section for discussing remaining
threats.
*  The RQs in the paper seem to have emerged during the runtime of the
experiment, rather than being defined upfront, indicating a more
exploratory nature of the study. It may be more appropriate to
classify the research design as exploratory rather than
hypothesis-driven.

## Comments on Rigor
The research in itself seems largely sound, albeit not well justified.

## Comments on Relevance
Relevance to software engineering (or even the fairness literature) is
rather unclear.

## Comments on Novelty
The paper explores a statistical relationship between two metrics in
depth that is expected to be there by construction. It may be novel in
a narrow sense, but contributes little to the fairness discourse.

The paper is not well grounded in the discourse on fairness in machine
learning or software engineering. It has a very narrow view of
fairness and discusses (noncritically) almost only related work from
software engineering venues. It seems to confuse various concepts in
this space (fairness of labels vs fairness of models, what it means to
be fair, …)

Basic fairness metrics are well established at this point and the
community has recognized that just measurement is not sufficient, when
it is even unclear which of multiple measurements are appropriate for
a problem. The fairness discourse in the community has moved far
beyond mere measurement. We would strongly encourage the authors to
engage more broadly with the fairness discourse outside of the
software engineering literature, e.g., reading papers in the ML, CHI
and FAccT communities or starting with the various books on this topic
published these days. The paper uses methods and datasets that would
be considered introductory examples on the topic.

## Comments on Verifiability and Transparency
No concerns.

## Comments on Presentation
Aside from many misleading statements on fairness concepts, the paper
is generally easy to read.

The paper is frustrating though in its use of citations that are often
not clearly mapped to statements in the paper. Several examples of
this are mentioned in minor comments above.

# Review 509B

Overall merit
3. Weak Accept (May accept but needs changes)

## Paper summary
The paper presents the results of an exploratory empirical experiment
comparing testing for fairness of the data vs the trained models. The
results are shortly discussed to confirm the correlation between data
and data-trained models with respect to group fairness; and the
implications for training data size and test reduction.

## Overall Comments for authors
The paper is easy to read and the work is interesting albeit
preliminary. I would have liked to learn more about (1) the
experimental setup, and (2) the suggestions to the software engineers
of ML systems, e.g. for how to use the visualizations in fig. 12 to
tune model training. As there is some space left (a bit more than half
page) maybe the authors could add some more concrete suggestions.

## Artifact Assessment
3. Average (overall acceptable though certain aspects may be missing)

## Comments on Rigor

The experiment design and execution appear to be well done. In spite
of its relatively small size, the experiment is interesting and
promising. The main limitations are helpful for possible extensions.

## Comments on Relevance
I find the work providing a good foundations for further research that
both optimizes model training and data collection. This is relevant
for SE for ML systems. Given the generalize explosion in size of both
datasets and ML models, this work may suggest pragmatic ways to
decrease both opportunistically.

## Comments on Novelty
To my knowledge the work is new. I am not an expert in ML hence
I cannot fully assess the claim that the state of the art all performs
fairness evaluation after the training. The literature discussion,
however, is interesting and clear. Also, I find the implications
concrete enough to inspire further research. The concrete suggestions
for the ML lifecycle (cf. fig. 1) could have been discussed more (they
are promised in section 2 but not further elaborated in section 5).

## Comments on Verifiability and Transparency
The replication package is provided. It however carries the text
"Replication package for paper titled Data vs. Model Machine Learning
Fairness Testing: An Empirical Study submitted to IJCAI 2023 main
track." This could be a concern: was this a double submission? and as
the notification of IJCAI was April 19, 2023, is is accepted there?

## Comments on Presentation The work is well structured, written
clearly and carrying sufficient information for both experiment
design, and description of results and implications. There are a few
typos (see below). Also, personally I missed the definition of 'data
drift' and 'fairness bug'. Some more explanations (mentioned before)
would be welcome (and there is some space in the paper). Some figures
are really too small, maybe to be mitigated with online figs in the
replication package combined with an explicit disclaimer/reference in
the first small figure like fig. 3.

A few typo's I could spot:
- noun 'practise' should be 'practice' I think consistently throughout
the paper; - 'where as' should be 'whereas'
- on page 1 RQ1: 'changes' should be 'change
- on page 2 sect. 2.0: 'used of' should be 'use of'
- sect. 5.3: 'cycle.If' should be 'cycle. If'
- sect. 5.4: 'pose' should be 'poses'
- sect 6.0: 'Section 4 and 5' should be 'Sections 4 and 5'
- references: I suggest double-checking all upper cases (e.g. in [13]
'ai' should be 'AI')

# Review 509C

Overall merit
1. Reject

## Paper summary

The paper explores fairness metrics prior to model training that can
be used to quantify bias in training data (DFM) and also fairness
metrics to quantify bias in the prediction of the trained model
(MFM). The paper conducts a study to understand relationship between
DFM and MFM for changes in the training dataset (sample size, feature
size). The authors found DFM and MFM are positively correlated and
convey the same information as the distribution and consequently the
fairness properties of the underlying training dataset
changes. Additionally, The training sample size has a profound effect
on the relationship between the DFM and MFM. Finally, DFM and MFM
convey similar information as the training sample size changes but not
when the feature sample size changes.

## Overall Comments for authors

The paper tackles an important problem of addressing bias
quantification in the training data. However, the methodology and
study investigate fairness and bias with highly simplified
assumptions. In Section 3.1, clearly define the fairness metrics used
on DFM and MFM. This is a core part of the contribution but there is
no description of these metrics or justification on their
selection. Favourable and unfavourable outcomes are represented by
0 and 1, and so is privileged and unprivileged groups. This seems like
oversimplification and a justification of why each protected feature
is not treated separately should be provided. Definition of favourable
versus unfavourable outcome needs to be provided. The authors simulate
a change in distribution of the training data by using a subset of the
training data. It is not clear why this is the best approach to
simulate a change in distribution. Why not perturb the data in
specific ways? The change in distribution is not verified. The study
primarily focuses on understanding the relationship between DFM and
MFM, why is this useful is not completely clear. Isn't the goal to
quantify bias both before and after training? So these also need to be
assessed individually in how well they address bias, and not just
their relationship.  There's a simple correlation analysis performed
in the empirical study but the value of the results and analysis of
reasons is not provided. Reason provided is typically training set
size or feature size which is the change initiated to begin with. In
Figure 6, the authors state they primarily observe a positive
correlation between the DFM and MFM. However, the correlation is very
weak with low values for the correlation coefficient. This should be
further discussed. How are features related to privileged or
unprivileged?

## Comments on Rigor
The study and approach lacks rigor. The study uses simplifying
assumptions that are not useful and there is lack of adequate
discussion of the results. More details provided in the comments for
authors.

## Comments on Relevance
The problem being tackled is relevant and timely.

## Comments on Novelty
Novelty is not clear as there is no discussion of the fairness metrics used.

## Comments on Presentation
The presentation lacks details and a substantial contribution.

# Review 509D

## Overall Comments for authors
Thank you for submitting your work to ICSE! The reviewers acknowledge
that this paper is from an important field of study. However, the
reviewers also identify important points for improvement. Among them,
the approach is lacking in depth, and the overall contribution to
software engineering as opposed to data science should be discussed
and made explicit. Also, the discussion of study design and results
need much more details, and maybe related to this, the novelty is also
not clear. The reviewers also suggest that further consulting the ML
fairness literature beyond SE venues would be significantly
beneficial. In conclusion, the reviewers unanimously agreed that the
manuscript is not ready for publication yet, and hope that the
constructive feedback provided by the reviewers will be beneficial to
the authors.
