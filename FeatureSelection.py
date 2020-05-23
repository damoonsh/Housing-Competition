
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
def tryOut(features, n, features_in=[]):
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
def polynomial_options(df, features, y_feature, threshhold=0.011):

    for feature in features:
        df[feature + '^2'] = df[feature] ** 2
        df[feature + '^3'] = df[feature] ** 3
        df[feature + '^4'] = df[feature] ** 4

        relations = df.corr()[y_feature]
        base = relations[feature]

        if not (relations[feature + '^4'] - base > 0.11 or relations[feature + '^4'] - relations[feature + '^3'] > threshhold  or relations[feature + '^4'] - relations[feature + '^2'] > threshhold):
            df.drop(feature + '^4', axis=1)
        if not (relations[feature + '^3'] - base > 0.11 or relations[feature + '^3'] - relations[feature + '^2'] > threshhold):
            df.drop(feature + '^3', axis=1)
        if not (relations[feature + '^2'] - base > threshhold):
            df.drop(feature + '^2', axis=1)
    return df