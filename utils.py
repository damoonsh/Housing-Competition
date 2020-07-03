# house/utils.py: some functions that help and increase the productivity the 
# preprocessing and fitting process written specific to the housing competition dataset
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
        Loads train and test datasets and puts into a dictionary and returns the splitted data based on their relative attributes in a dictionary
        
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

def combine_train_test(train, test, y_feat='SalePrice'):
    """ Returns a combined version of the train and test datasets. """
    train.drop([y_feat], axis=1 , inplace = True) # Drop the dependent column in traininig data
    feat_cols = train.append(test) # Combine datasets
    feat_cols.reset_index(inplace=True) # Reset Indexes
    feat_cols.drop(['index', 'Id'], inplace=True, axis=1) # Drop Id and index columns

    return feat_cols


def missing_info(data):
    """ Returns two dataframes (train, test) defining their relative missing values. """
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


def validate(y_pred):
    """ Prints out the data validation with respect to the highest submissions. """
    from sklearn.metrics import mean_absolute_error as MAE
    from sklearn.metrics import mean_squared_log_error as MSLE
    # Import the base_validation submititions
    b012 = load_bench_data(file_name='012008.csv', root='./submissions/')['SalePrice']
    b011 = load_bench_data(file_name='011978.csv', root='./submissions/')['SalePrice']
    
    # Print out the differences
    print('MAEs:')
    print('b011:', int(MAE(b011, y_pred)) / 1000)
    print('b012:', int(MAE(b012, y_pred)) / 1000)
    print('-----------------------------------')
    print('base-differences:', int(MAE(b011, b012)) / 1000)
    print('###############################################')
    print('Lograithmic Error:')
    print('MSLE:')
    print('b011:', MSLE(b011, y_pred))
    print('b012:', MSLE(b012, y_pred))
    print('-----------------------------------')
    print('base-differences:', MSLE(b011, b012))


def quantize(values):
    """ Emits the floating point in a list of numbers. """
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
 
# Implement the outliers utility
def encode_categorical_feature(df, category, y_feature='SalePrice', outlier=False,  include_nan=True):
    """
        Given that there categorical variables, we want to have them ranked based on their value

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
            
        # Returns:
            imputed column values with the encoding dictionary
            True if the data needed raking False if not
    """
    vals_list = list(df[category].unique())
    
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
    
    if not include_nan:
        haveNan = False
        means[np.nan] = 0

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
    
    for cat in unique_categories:
        if cat == 'nan':
            means[np.nan] = round(means[np.nan] / AVG, 4)
        else:
            means[cat] = round(means[cat] / AVG, 4)
    
    # IF the Type was not softmax return averages
    return means

# Implement: give a list of features which have more than 10% missing data but we still want to include them
def get_encoding_dicts(df, features):
    """ Returns the dictionary containing the encoded values for each unique value inside the categorical columns. """
    cat_dicts = {}
    len_df = df.shape[0]
    
    for feature in features:
        if df[feature].isna().sum() / len_df < 0.1:
            cat_dicts[feature] = encode_categorical_feature(df, feature)
        else:
            print('Ignoring:', feature)
            cat_dicts[feature] = encode_categorical_feature(df, feature, include_nan=False)
    
    return cat_dicts


def encode_categorical(df, cat_dicts):
    """ Encodes the dataframe's categorical features by mapping them to their relative dictionary values. """
    
    for feature in cat_dicts.keys():
        df[feature] = df[feature].map(cat_dicts[feature])
    
    return df


def normalize(col, Type='std'):
    """
        Normalizes the given column of data
        
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
