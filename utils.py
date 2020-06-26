# house/utils.py: some functions that help and increase the productivity the preprocessing and fitting process
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


def load_bench_data(file_name, root='./submissions/'):
    """ Loads and formats the benchmark data """
    bench = load_data(file_name ,root)
    bench['Id'] = bench['Id'] - bench.shape[0] - 1 
    bench = bench.set_index('Id')

    return bench


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


def missing_info(data):
    """ retuns two dataframes (train, test) defining their relative missing values. """
    # test data:
    cat_dict = {
        "Test": dict(data['test'][data['test_cat_missing']].isna().sum()), 
        "Train": dict(data['train'][data['train_cat_missing']].isna().sum())
    }

    # train data:
    num_dict = {
        "Test": dict(data['train'][data['train_num_missing']].isna().sum()), 
        "Train": dict(data['test'][data['test_num_missing']].isna().sum())
    }

    return pd.DataFrame(cat_dict).fillna(0), pd.DataFrame(num_dict).fillna(0)


def quantize(values):
    """ emits the floating point in a list of numbers. """
    modified = [] 
    for num in list(values):
        if num - int(num) >= 0.5:
            modified.append(int(num) + 1)
        else:
            modified.append(int(num))
            
    return modified


def getNaIndexes(df, feature=''):
    """ Gets the index of columns with nan property. """ 
    return list(df[feature].index[df[feature].apply(np.isnan)])


def stringify_keys(l):
    """ casts all the values as str """
    for item in l:
        if type(item) != str:
            l.remove(item)
            l.append(str(item))
    return l
 

def encode_categorical_feature(df, category, y_feature='SalePrice', Type='average', outlier=False, uniques=None):
    """
        Given that there categorical variables, we 
        want to have them ranked based on their value
        
        # Arguments:
            df: Dataframe
            category: the category (feature) to be imputed
            y_feature: the independent feature that we base our
                ranking on
            Type: 
                'average' would average the values, 
                'norm' returns the normalized version of means
            outlier: given that it is set to True, the outliers in the 
                y_feature of the dataframe would not be considered
            uniques: Sending 
            
        # Returns:
            imputed column values with the encoding dictionary
            True if the data needed raking False if not
    """
    if uniques is None:
        # Getting the unique values for the column
        vals_list = list(df[category].unique())
    else:
        vals_list = uniques
    
    unique_categories = stringify_keys(vals_list)
    haveNan = False # Check to see if there is na/nan in unique vales
    
    # Deleting NaNs since they are going to be considered seperately
    if 'nan' in unique_categories:
        haveNan = True
        i = unique_categories.index('nan')
        unique_categories.pop(i)
        
    # Dictionary containing mean values of different values in column
    means = {}
    AVG = 0 # Sum of all averages
    
    # Going through unique values
    for cat in unique_categories:
        cat_avg = df.loc[df[category] == cat][y_feature].mean()
        means[cat] = cat_avg
        AVG += cat_avg
        
    # Now considering the nan's or the values that were not in any of the unique
    if haveNan:
        na_avg = df.loc[~df[category].isin(unique_categories)][y_feature].mean()
        means[np.nan] = na_avg
        AVG += na_avg
        unique_categories.append('nan')
    
    if Type == 'plain':
        return means
    
    for cat in unique_categories:
        if cat == 'nan':
            means[np.nan] /= AVG
        else:
            means[cat] /= AVG
    
    # IF the Type was not softmax return averages
    return means


def map_categorical_dicts(df, cat_dicts):
    """ Map the dictionary in the feature of the dataset."""
    features = list(cat_dicts.keys())

    for feature in features:
        df[feature] = df[feature].map(cat_dicts[feature])
        
    return df


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
    
    for feature in features:
        # get the dictionary of averages
        feat_dic = encode_categorical_feature(df, feature, y_feature, Type)
        # Change the values based on their corresponding value
        col = df[feature].copy()
        df[feature] = df[feature].map(feat_dic)
		
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
