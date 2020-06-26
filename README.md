# Housing competition

This repository is the solution (ZLAS team) for [Kaggle's housing competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
current-rank: 802

## Preprocessing

The general strategy is to combine both the categorical and numerical values in the training and testing and then process them at the same time. For categorical variables we will be getting dictionaries from the training data and then process them for the combined dataframe. In the case of numerical features, imputation is going to be done using the KNN imputation.

1. Encoding categorical features: I'll be using some functions written in utils.py to come up with meaning values for the unique keys in each of the categorical features, then map them in the data given certain conditions.
2. Imputing numerical data: The numerical

### In-depth analysis of categorical variables

1. Compare the different NaNs for the same categories (and not) in the number of NaNs they have.
2. Given that 90% data is not missing for a given feature (column) map their encoded numerical values in the dataframe, otherwise, only impute non-nan values in the feature and then impute the rest of the missing values using any other technique. Dropping the column for values with too many missing might be a general option but in order to use the data for Neural Networks, it would make sense to just impute the missing values with zeros.

### Encoding categorical variables

In order to come up with a meaningful value for any given unique value in a categorical feature column, we will be considering the average SalePrice for each of those unique values and weight them relative to each other. The important thing to note would be that given that more than 90% exists in a column we could just impute the minor missing values with the average of SalePrice for those columns. But if less then 90% of the data existed then there would be a problem since our measures would not make sense and since we are using Neural Networks it would make more sense to impute them with zeros.

## Model:

Some Assumptions about the models:

1. The evaluation metric used for the competition is Mean Squared Logarithmic Error, hence all the models are going to be using the MSLE as their evaluation metric.
2. The Default optimizer will be Adam and RSMprop, also we could choose to use a learning_rate modifier for the Optimizer which would either be a InverseTimeDecay or ExponentialDecay.
3. They are only few type of layers and activation that work reasonably for Regression problems:
    Layers: Dense, Dropout, BatchNormalization
    activations: relu, elu, tanh*

4. For each model, there should be a config dictionary that describes values that best work for the model and this values should have been driven by testing and parameter tuning (potentially using Keras-tuner).
5. Each function that implements the model gets a full configuration on both compilation and fitting parameters that the model going to need.
6. The monitored parameter in a EarlyStopping object is always 'val_loss', since the assumption is that we will be using validation_data in order to validate our data

Note01: BatchNormalization is a better alternative to normalizing data. additionally, BatchNormalization layers should be used only at the beginning layer of the model since it does not work when used in different parts of the model.
Note02: tanh does not happen to work as well as elu and relu, on a broader perspective relu is  most appropriate one for the regression.
