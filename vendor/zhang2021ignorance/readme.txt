This package includes the code and extra results for ICSE2021 submission ``Ignorance and Prejudice'' in Software Fairness.

We took several steps to address the threats to construct validity. First, we use different machine learning models (e.g., Decision Trees, Logistic Regression, Random Forests, AdaBoost) to examine whether the chosen machine learning model is a factor that would affect our conclusions. Second, for the default Decision Trees model, we use different complexity configuration (i.e., different fixed maximum depths and also grid search for each feature set and training data) to check whether complexity is a factor that would affect our conclusions. Third, we try different orders when extending the feature set when answering RQ1. Our paper provides results with the default configuration with one model and one fixed complexity configuration. 

This repository contains the results for other models and configurations, together with our code. All extra results demonstrate that the default configuration is not a threat to our conclusions.


Below is the content inside each package.

code: the source code for our experiments
results_extramodels: the results with extra machine learning models, including LogisticRegression, AdaBoost, RandomForests
results_featureorder: the results with reversed feature order. 
results_differentconfiguration:
- gridsearch: the results with DecisionTree when the maximum_depth is configurd by grid search
- smaller: the results with DecisionTree when the maximum_depth is 5