import numpy as np
import pandas as pd


def load_data(fileName="train.csv", base_path="./dataset/"):
    """ 
        Loads the data into the variables 
        
        # Arguments:
            filename: name of the .csv file
            base_path: path to were the data is
        
        # Returns:
            returns the dataset in a np.array
    """
    path = base_path + fileName
    return pd.read_csv(path)


def load_bench_data():
    """ Loads and formats the benchmark data """
    bench = load_data('bench-mark.csv' ,'./dataset/')
    bench['Id'] = bench['Id'] - bench.shape[0] - 1 
    bench = bench.set_index('Id')

    return bench


def impute_numerical(data, Type='train'):
    # LotFrontage and GarageYrBlt should equal zeos 
    # for NaN's since it means 0 meters of lot frontage
    data['LotFrontage'] = data['LotFrontage'].fillna(0)
    data['MasVnrArea'] = data['MasVnrArea'].fillna(0)
    # If this is the training data then use knn
    if Type == 'train':
        data['GarageYrBlt'] = fillNaWithKNN('GarageYrBlt', data.copy(), 'SalePrice')
    
    return data


def preprocess_train(data):
    data = impute_numerical(data)
    data = encode_categorical(train, data.select_dtypes(['float64', 'int64']).columns[numerical_train.isna().any()].tolist())
    
    return data


def retrieve_data():
    """ 
        Loads train and test datasets and puts into a dictionary and returns 
        the splitted data based on their relative attributes in a dictionary

        further improvement:
        The datatypes within data could be identified and then
        classified based on the available data types
        
        # Returns:
            dictionary containing info on train and test data
     """
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
        'test_cat_missing': missing_cat_train, 'test_num_missing': missing_num_test,
        'train_cat_list': list(categorical_train.columns), 'train_num_list': list(numerical_train.columns)
    }

    return dictionary


def getNaIndexes(feature, df):
    """ 
        Gets the index of columns with nan property. 

        # Arguments:
            feature: feature of the dataframe
            df: dataframe

        # Returns:
            indices of na values in the 
    """
    return list(df[feature].index[df[feature].apply(np.isnan)])


def divideByNA(feature, l, df, y_feature='SalePrice'):
    """
        Divides the DataFrame into two parts: with na's and without na's

        # Arguments: 
            feature: target feature which data will be splitted based on
            l: list of indices with na values
            y_featrue: in this function this is the independent value
            
        # Returns:
            X, y, X_test
    """
    # X is the y_feature for the non-missing values
    # y is the feature for non-missing values
    # X_test is the y_feature for missing values
    X, y, X_test = [], [], []
    
    for i in range(0, df.shape[0]):
        if i in l:
            X_test.append(df.iloc[i][y_feature])
        else:
            y.append(df.iloc[i][feature])
            X.append(df.iloc[i][y_feature])

    return np.reshape(X, (-1, 1)),  np.reshape(y, (-1, 1)),  np.reshape(X_test, (-1, 1))


def fillNaWithKNN(feature, df, y_feature):
    """
        Impute the missing data with KNN method
    
        # Arguments:
            feature: feature in the dataframe to impute
            df: Dataframe
            y_feature: independent feature
        
        # Returns:
            column with imputed data
    """
    # If there were no missing values
    if not df[feature].isna().any():
        return df[feature]
    
    from sklearn.neighbors import KNeighborsRegressor
    # Instantiate a KNN model
    number_of_neighbors = df.shape[0] // 2 + 1 # Get the number of neighbors to compare with
    knn = KNeighborsRegressor(n_neighbors=number_of_neighbors, weights='distance')
    # Get the na indexes
    na_indexes = list(df[feature].index[df[feature].apply(np.isnan)])
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


def stringify_keys(l):
    """
        casts all the values as str
        
        # Arguments:
            l: list
        # Returns:
            new list with all str values
    """
    for item in l:
        if type(item) != str:
            l.remove(item)
            l.append(str(item))
    return l
 

def rank_categorical_values(df, category, y_feature='SalePrice', Type='average', outlier=False):
    """
        Given that there categorical variables, we 
        want to have them ranked based on their value
        
        # Arguments:
            df: Dataframe
            category: the category (feature) to be imputed
            y_feature: the independent feature that we base our
                ranking on
            Type: 'average' would average the values, 'softmax' would
                perform softmax on the values
            outlier: given that it is set to True, the outliers in the 
                y_feature of the dataframe would not be considered
            
        # Returns:
            imputed column values with the encoding dictionary
            True if the data needed raking False if not
    """
    # Getting the unique values for the column
    vals_list = list(df[category].unique())
    
    # If the data type in the first one is not either na or str then it is a number
    # And won't need processing
    if not (pd.isna(vals_list[0]) or type(vals_list[0]) == str):
        print('Did not need rankking')
        return {}, False
    unique_categories = stringify_keys(vals_list)
    haveNan = False # Check to see if there is na/nan in unique vales
    
    # Deleting nans since they are going to be considered seperately
    if 'nan' in unique_categories:
#         print('has nan')
        haveNan = True
        i = unique_categories.index('nan')
        unique_categories.pop(i)
        
    # Dictionary containing mean values of different values in column
    means = {}
    AVG = 0 # Sum of all averages
    
    # Going through 
    for cat in unique_categories:
        cat_avg = df.loc[df[category] == cat][y_feature].mean()
        means[cat] = cat_avg
        AVG += cat_avg
        
    # Now considering the nan's or the values that were not in any of the unique
    if haveNan:
        na_avg = df.loc[~df[category].isin(unique_categories)][y_feature].mean()
        means['nan'] = na_avg
#         print('avg: ',na_avg)
        AVG += na_avg
        unique_categories.append('nan')
    
    for cat in unique_categories:
        means[cat] /= AVG
#     print(means)
    if Type == 'softmax':
        return softmax(means), True
        
    # IF the Type was not softmax return averages
    return means, True


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
        
    return col.astype(float)


def encode_categorical(df, features, y_feature='SalePrice', Type='average'):
    """
        Encodes the dataframe's categorical features 
    
        # Argument:
            df: Dataframe (NOTE: pass df.copy() for explicit mutation)
            featrues: Categorical features to be encoded
            y_feature: independent data used base our measures on
            
        # Returns:
            encoded data
    """
    # Dictionary containing all the rankings
    dic = {}
    dics = {}
    
    for feature in features:
        # get the dictionary of averages
        feat_dic, change = rank_categorical_values(df, feature, y_feature, Type)
        # IF the feature did not need any ranking: pass
        if not change: pass
        # Change the values based on their corresponding value
        col = df[feature].copy()
        df[feature] = impute_rank_weight(col, feat_dic)
#         print(df[feature])
    
    return df


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
    q1 = df[feature].quntile(0.25)
    q2 = df[feature].quntile(0.5)
    q3 = df[feature].quntile(0.75)
    iqr = q3 - q1
    upper_bound = q1 - 1.5 * iqr
    lower_bound = q3 + 1.5 * iqr
    
    df[feature] = df[feature][df[feature] < upper_bound]
    df[feature] = df[feature][df[feature] < lower_bound]
    
    return df

def normalize(col, Type='std'):
    """
        Normalizes the 
        
        # Arguments:
            col: column of data
            Type: avg scales the data with average of column
                  std scales the data with standard deviation of the column
    
        # Returns:
            returns the normalized data
    """
    if Type == 'avg':
        return (col - col.mean()) / col.mean()
    
    return (col - col.mean()) / col.std()


def build_model01():
  model = keras.Sequential([
    layers.Dense(64, input_shape=[len(std_d.keys())]),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='softmax'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='softmax'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='softmax'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.01)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mse', 'msle'])
  return model