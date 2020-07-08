# Housing competition

This repository is the solution (ZLAS team) for [Kaggle's housing competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
current-rank: 943

## Preprocessing

The general strategy is to combine both the categorical and numerical values in the training and testing and then process them at the same time. For categorical variables we will be getting dictionaries from the training data and then process them for the combined dataframe. In the case of numerical features, imputation is going to be done using the KNN imputation.

1. Encoding categorical features: The goal is to find a number that could represent the unique values in each of the categorical columns. By having those values each of the string values will be replaced with a numerical one. The encode_categorical_feature function in ./utils.py gets the dataset and the column that we are trying to encode, then it returns the average of the all prices with a given value in the column divided by the sum of all averages for different values in the column: Given that x1,x2,x3,..,xn are unique values in column C, the average SalePrice (dependent column in the dataset) when C is x1 will be avg1, and respectively for each of these unique values in the dataset, there will be avg2, avg3, ..., avgn. Now, in order to have small values for encoding each of the averages will be divided by the sum of all averages: Avg1 = avg1 / (avg1 + 1vg2 + ..+ avgn) and so on. As it is apparent the sum of all the returned encoding values will be one: Avg1 + Avg2 + ... + Avgn = 1. Now when using this technique we can impute the NaN values within the categorical features just by replacing them with the same logic and it will not need any further imputation. Yet it is important to note that if the number of missing values (NaN) is a lot then we should not do the encoding for NaNs and impute them after they are encoded. So, if more than 10% if the data was missing then don't encode the Nan in those columns (implemented in get_encoding_dicts).
2. Imputing numerical data: The numerical data will simply be imputed using the KNN imputer module of sklearn.

### In-depth analysis of categorical variables

1. Compare the different NaNs for the same categories (and not) in the number of NaNs they have.
2. Given that 90% data is not missing for a given feature (column) map their encoded numerical values in the dataframe, otherwise, only impute non-nan values in the feature and then impute the rest of the missing values using any other technique. Dropping the column for values with too many missing might be a general option but in order to use the data for Neural Networks, it would make sense to just impute the missing values with zeros.

## Model

Some Assumptions about the models:

1. The evaluation metric used for the competition is Mean Squared Logarithmic Error, hence all the models are going to be using the MSLE for their evaluation metric.
2. The Default optimizer will be Adam and RSMprop (for different models), also we could choose to use a learning_rate modifier for the Optimizer which would either be a InverseTimeDecay or ExponentialDecay.
3. Dense, Dropout, and BatchNormalization are the main layers used for the models. And relu and elu are the most effective activation functions used for the regression problems.
4. The monitored parameter in a EarlyStopping object is always 'val_loss', since the assumption is that we will be using validation_data in order to validate our data and prevent over-fitting.

The goal is to have a number of models predict a set of values and then average them based on their accuracy level. The general workflow is to construct a Neural Network and then fine tune it (either manually or by keras-tuner), and given that a model had high accuracy level, it will be saved in models folder and will be loaded further along the way to make predictions. 