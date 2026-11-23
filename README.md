# Credit Card Fraud Detection under Extreme Class Imbalance

CS5998 Capstone Project — A A L N Perera (268476J)

## Overview

Only 0.172% of the transactions in the dataset are fraud, so a model that always predicts
"not fraud" gets about 99.8% accuracy and catches nothing. This project builds a reproducible
pipeline that treats fraud detection as both a classification and an outlier-detection
problem, and judges models by precision, recall, F1 and PR-AUC instead of accuracy.

- **Baseline:** Logistic Regression
- **Improved:** Random Forest with class weights, SMOTE and decision-threshold tuning
- **Comparison:** Isolation Forest (unsupervised outlier detection)

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(ULB Machine Learning Group, Kaggle): 284,807 transactions, 492 fraud. The raw data is not
stored in this repository. A download step will be added.

## Repository structure

```
data/            raw and processed data (not tracked by git)
notebooks/       exploratory analysis
src/             pipeline code (data, features, models, evaluation)
scripts/         command-line entry points
results/         generated figures and tables
report/          milestone reports
```

## Status

Work in progress.
