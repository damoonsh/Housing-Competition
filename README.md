# Housing competition
This repository is my attempt of [Kaggle's housing competiton](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

## Data Preprocessing:
1. They were some columns with a large number missing values, and those features were not significant. So, at the beginning all of those values were droped from the data frame.

2. The missing numerical data were imputed by using KNNRegressor estimator of sklearn.
    Using features that are not missing to imputate the missing data with k-means algorithm
		NOTE[1]: Given that there are small fraction of missing values in the a feature
		, we can first impute that by filling it with mean or mode.

		K-means clustering:
			x1, x2, x3, x4, x5 are the features where x5 has missing values
			now there is going to be a k-means ran on each of the features x1-4
			hence each data point in each feature will be assigned a group and we 
			are interested in that group's average. Then in order to predict the
			value of the missing data, each feature will have a weight and a linear
			-wise calculation will indicate the missing value:
			1. Running k-means on each x1-4 feature (non-missing) and obtaining
			   each of their averages.
			2. Going over the missing value point where x5(i) is missing but we have
			   x1-4(i) values, and we also know the features correlation with each other
			   (with .corr() functionality) which will be called w15,w25,w35,w45 then we
			   know the average of each group that each data point features are in g1,g2,
			   g3,g4, hence: x5(i) = w15 * g1 + ... + w45 * g4

			   g denotes the avergae of that cluster,
			   w denotes weights where w15 = rel15 / (rel15 + ... + rel45)

## Model:

* At first the features were selected by running multiple Linear Regression on different combination of features.
* Running SGD regression on the data and making the final predictions.