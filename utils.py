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

def softmax(dic):
    """
        Performs softmax on a set of values
        
        # Arguments:
            dic: dictionary of values
            
        # Returns:
            A softmax version of the given dictionary
    """
    avg = 0
    for val in dic.keys():
        avg += np.exp(dic[val])
        dic[val] = np.exp(dic[val])
        
    for val in dic.keys():
        dic[val] /= avg
    
    return dic

def rank_categorical_values(df, category, y_feature='SalePrice', Type='average'):
    """
        Given that there categorical variables, we 
        want to have them ranked based on their value
        
        # Arguments:
            df: Dataframe
            category: the category (feature) to be imputated
            y_feature: the independent feature that we base our
                ranking on
            Type: 'average' would average the values, 'softmax' would
                perform softmax on the values
        
        # Returns:
            imputated column values with the encoding dictionary
    """
    # Getting the unique values for the column
    unique_categories = list(df[category].unique())
    haveNan = False # Check to see if there is na/nan in unique vales
    
    # Deleting nans since they are going to be considered seperately
    if str(unique_categories[0]) == 'nan':
        haveNan = True
        i = unique_categories.index(np.nan)
        unique_categories.pop(i)
    # Dictionary containing mean values of different values in column
    means = {}
    AVG = 0 # Sum of all avergaes
    
    # Going through 
    for cat in unique_categories:
        cat_avg = df.loc[df[category] == cat][y_feature].mean()
        means[cat] = cat_avg
        AVG += cat_avg
        
    # Now considering the nan's or the values that were not in any of the unqiue
    if haveNan:
        na_avg = df.loc[~df[category].isin(unique_categories)][y_feature].mean()
        means['nan'] = na_avg
        AVG += na_avg
        unique_categories.append('nan')
    
    for cat in unique_categories:
        means[cat] /= AVG
        
    if Type == 'softmax':
        softmax(means)
        
    return means

def impute_rank_weight(col, dic):
    """
        Imputes the str values with numbers which are
        their relative weights.
        
        # Arguments:
            col: a copied version of a slice of the data
            dic: Dictionary that contains the averages of the 
                 given unique values in the column
                 
        # Returns:
            decoded column
    """
    unique_values = dic.keys()
    
    for val in unique_values:
        col.loc[col == val] = dic[val]
        
    if 'nan' in  unique_values: 
        col.loc[pd.isna(col)] = dic['nan']
        
    return col

def encode_categorical(df, features, y_feature='SalePrice'):
    """
        Encodes the dataframe's categorical features 
    
        # Argument:
            df: Dataframe (NOTE: pass df.copy() for explicit mutation)
            featrues: Categorical features to be encoded
            y_feature: independent data used base our measures on
            
        # Returns:
            encoded data
    """
    # Get a dictionary containing all the rankings
    dic = {}
    
    for feature in features:
        
        # get the dictionary of averages
        feat_dic = rank_categorical_values(df, feature)
        # Change the values based on their corresponding value
        col = df[feature].copy()
        df[feature] = impute_rank_weight(col, feat_dic)
    
    return df

# Implement this function
def emit_outliers(df, feature):
    """
        Deletes the outliers in a column so the modeling would be 
        more accurate.
        
        # Arguments:
            df: Dataframe
            feature: feature that is to be inspected
        
        # Returns:
            A dataframe with deleted outliers
    """
    pass