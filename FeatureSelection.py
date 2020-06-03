import pandas as pd
import numpy as np
"""
    Functions below are the ones used for feature selection
"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score

""" 
    Generates a list of features with the choice of k  out of n. 
"""
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
def tryOut(features, n, train, features_in=[], y_feature='SalePrice'):
    # Getting the various feature combinations
    combs = f(features, n)
    # Setting the initial index and dict to null
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

"""
    Adding polynomials to the features
"""
def polynomial_options(df, features, max_exponent=3,threshhold=0.011):
    for feature in features:
        for exponent in range(1,max_exponent+1):
            # Get the exponent 2, 3, 4 of the features
            df[feature + '^2'] = df[feature] ** exponent
        
    # Returning the new df
    return df