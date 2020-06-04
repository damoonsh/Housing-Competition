import numpy as np
import pandas as pd

"""
    Loads the data into the variables
"""
def load_data(fileName="train.csv", base_path="./dataset/"):
    path = base_path + fileName
    return pd.read_csv(path)

""" Loads and formats the benchmark data"""
def load_bench_data():
    bench = load_data('bench-mark.csv' ,'./')
    bench['Id'] = bench['Id'] - bench.shape[0] - 1 
    bench = bench.set_index('Id')

    return bench

""" 
    Retrieve Data:
    gets the different datasets and returns the splitted data based on
    their relative attributes in a dictionary

    further improvement:
    The datatypes within data could be identified and then
    classified based on the available data types
 """
def retrieve_data():
    # Load Data
    test = load_data(fileName='test.csv')
    train = load_data(fileName='train.csv')

    # Categorical Data
    categorical_train = train.select_dtypes('object')
    numerical_train = train.select_dtypes(['float64', 'int64'])
    # Numerical Data
    categorical_test = test.select_dtypes('object')
    numerical_test = test.select_dtypes(['float64', 'int64'])

    # Missing data 
    missing_cat_train = categorical_train.columns[categorical_train.isna().any()].tolist()
    missing_cat_test = categorical_test.columns[categorical_test.isna().any()].tolist()
    missing_num_train = numerical_train.columns[numerical_train.isna().any()].tolist()
    missing_num_test = numerical_test.columns[numerical_test.isna().any()].tolist()
    
    # Dictionary containing the needed information
    dictionary = {
        'train': train, 'test': test,
        'train_cat': categorical_train, 'test_cat': categorical_test, 
        'train_num': numerical_train, 'test_num': numerical_test,
        'train_cat_missing': missing_cat_train, 'train_num_missing': missing_num_train,
        'test_cat_missing': missing_cat_train, 'test_num_missing': missing_num_test
    }

    # Return dictionary
    return dictionary
"""

"""
def normalize(df, type='mean'):
    pass
"""
    Preprocesses the data so it would be ready to fit into the model
"""
def Preprocess(data):
    # Droping All the columns with more than half of rows NaNs simply because 
    # they might not be as important as the other columns
    data.dropna(thresh=data.shape[0] / 2, axis=1, inplace=True)


"""
    Iteratively goes through various combination of features and selects the best ones.
"""
def FeatureSelection(features_range=10, choice_number=6, handPickedFeatures = []):
    pass

"""
    helper functions
"""

""" Gets the index of columns with nan property. """
def getNaIndexes(feature, df):
    return list(df[feature].index[df[feature].apply(np.isnan)])

"""
    Divides the DataFrame into two parts:
    with na's and without na's

    @params: 
        X is the y_feature for the non-missing values
        y is the feature for non-missing values
        X_test is the y_feature for missing values
"""
def divideByNA(feature, l, df, y_feature='SalePrice'):
    X, y, X_test = [], [], []
    
    for i in range(0, df.shape[0]):
        if i in l:
            X_test.append(df.iloc[i][y_feature])
        else:
            y.append(df.iloc[i][feature])
            X.append(df.iloc[i][y_feature])

    return np.reshape(X, (-1, 1)),  np.reshape(y, (-1, 1)),  np.reshape(X_test, (-1, 1))

"""
    Impute the missing data with KNN method
"""
def fillNaWithKNN(feature, df, y_feature):
    from sklearn.neighbors import KNeighborsRegressor
    # Instantiate a KNN model
    number_of_neighbors = df.shape[0] // 2 + 1 # Get the number of neighbors to compare with
    knn = KNeighborsRegressor(n_neighbors=number_of_neighbors, weights='distance')
    # Get the na indexes
    na_indexes = getNaIndexes(feature, df)
    # Split the data based on the na
    X, y, X_test = divideByNA(feature, na_indexes, df, y_feature)
    # Fit the data
    knn.fit(X, y.ravel())
    # Make the prediction
    y_test = knn.predict(X_test)

    # Apply the new values
    c = list(df[feature].copy())
    index = 0
    
    for i in range(df.shape[0]):
        if i in na_indexes:
            c[i] = y[index][0]
            index += 1
    
    df[feature] = c
    
    return df[feature]