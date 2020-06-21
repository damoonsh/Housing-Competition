from utils import getNaIndexes
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

class KKNImputator:
    """
         This module will impute data using one indepdent variable 
         in the dataset.
    """
    def __init__(self, data, X, y):
        """
            # Arguments:
                data: dataframe containing the data
                X: independent feature(s) in data that will be used to impute data
                y: dependent feature with missing value in the data
        """
        self.data = data
        # Initialize a KNeighborsRegressor with half + 1 of the data
        self.model = KNeighborsRegressor(self.data.shape[0] // 2 + 1, weights='distance')
        self.X = X
        self.y = y
        self.missing_indices = getNaIndexes(X, data)

    
    def divideByNA(l):
        """
            Divides the DataFrame into two parts: with na's and without na's

            # Returns:
                X, y, X_test
        """
        # X is the y_feature for the non-missing values
        # y is the feature for non-missing values
        # X_test is the y_feature for missing values
        X, y, X_test = [], [], []

        for i in range(0, self.data.shape[0]):
            if i in self.missing_indices:
                X_test.append(self.data.iloc[i][self.y])
            else:
                y.append(self.data.iloc[i][self.X])
                X.append(self.data.iloc[i][self.y])

        return np.reshape(X, (-1, 1)), np.reshape(y, (-1, 1)), np.reshape(X_test, (-1, 1))
    