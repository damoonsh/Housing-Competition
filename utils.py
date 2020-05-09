import numpy as np
import pandas as pd

"""
    Loads the data into the variables
"""
def load_house_data(fileName="train.csv", base_path="./dataset/"):
    path = base_path + fileName
    return pd.read_csv(path)

"""
    Preprocesses the data so it would be ready to fit into the model
"""
def Preprocess(data):
    # Droping All the columns with more than half of rows NaNs simply because 
    # they might not be as important as the other columns
    data.dropna(thresh=data.shape[0] / 2, axis=1, inplace=True)

    # Getting the features with missing values divifing the 
    # dataset into categorical and numerical
    categorical_data = data.select_dtypes('object')
    numerical_data = data.select_dtypes(['float64', 'int64'])

    # Get the columns with missing data
    missing_categorical_data = categorical_data.columns[categorical_data.isna().any()].tolist()
    missing_numerical_data = numerical_data.columns[numerical_data.isna().any()].tolist()

"""
    Iteratively goes through
"""
def FeatureSelection(features_range=10, choice_number=6, handPickedFeatures = []):

"""
    helper functions
"""
from sklearn.neighbors import KNeighborsClassifier

""" Gets the index of columns with nan property. """
def getNaIndexes(feature, df):
    return list(df[feature].index[df[feature].apply(np.isnan)])

"""
    Divides the DataFrame into two parts:
    with na's and without na's
"""
def divideByNA(feature, l, df, y_feature):
    X, y, X_test = [], [], []
    
    for i in range(0, df.shape[0]):
        if i in l:
            X_test.append(df.iloc[i][y_feature])
        else:
            X.append(df.iloc[i][feature])
            y.append(df.iloc[i][y_feature])
            
    return np.reshape(X, (-1, 1)),  np.reshape(y, (-1, 1)),  np.reshape(X_test, (-1, 1))

"""
    Fill the nas with knn method
"""
def fillNaWithKNN(feature, df, y_feature):
    # Instantiate a KNN model
    knn = KNeighborsClassifier(n_neighbors=df.shape[0]//2 + 1)
    # Get the na indexes
    l = getNaIndexes(feature, df)
    # Split the data based on the na
    X, y, X_test = divideByNA(feature, l, df, y_feature)
    # Fit the data
    knn.fit(X, y.ravel())
    # Make the prediction
    y_test = knn.predict(X_test)
    
    # Apply the new values
    c = list(df[feature].copy())
    index = 0
    
    for i in range(df.shape[0]):
        if i in l:
            c[i] = y[index]
            index += 1
    
    df[feature] = c
    
    return df[feature]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score

""" Generates a list of features with the choice of k  out of n. """
def f(l, k):
    n = len(l)
    d = []
    if k == 1 or k == n: return l
    Range = n - k + 1
    for i in range(0, Range):
        nl = f(l[i + 1:], k - 1)
        for z in nl:
            if k == len(nl) + 1:
                nl.append(l[i])
                d.append(nl)
                break
            if type(z) != list:
                d.append([z, l[i]])
            else:
                zz = z.copy()
                zz.append(l[i])
                d.append(zz)
    return d

""" 
    This function tries out the different combinations and 
    gives out the performance of each of them did.
"""
def tryOut(features, n, features_in=[]):
    # Getting the various feature combinations
    combs = f(features, n)
    # Setting the inital index and dict to null
    r2s = {}
    index  = 0
    # Instantiating the columns
    r2s['combination'], r2s['val1'], r2s['val2'], r2s['val3'], r2s['std'], r2s['r2'], r2s['mean'] = [], [], [], [], [], [], []
    
    for comb in combs:
        
        if features_in != []:
            comb.extend(features_in)
        
        reg = LinearRegression()
        X = train[comb]
        y = train[y_feature]
        
        r2s['combination'].append(comb)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        
        r2 = r2_score(y_pred, y_test)
        r2s['r2'].append(r2)
        
        d = list(cross_val_score(reg, X,  y, cv=3))
        
        r2s['val1'].append(d[0])
        r2s['val2'].append(d[1])
        r2s['val3'].append(d[2])
        r2s['mean'].append(np.mean(d))
        r2s['std'].append(np.std(d))
        
        index += 1
        
    return pd.DataFrame(r2s)