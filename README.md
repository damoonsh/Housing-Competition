# Housing competition
This repository is my attempt of [Kaggle's housing competiton](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

## Data Preprocessing:
1. They were some columns with a large number missing values, and those features were not significant. So, at the beginning all of those values were droped from the data frame.

2. The missing numerical data were imputed by using KNNRegressor estimator of sklearn.

## Model:

* At first the features were selected by running multiple Linear Regression on different combination of features.
* Running SGD regression on the data and making the final predictions.