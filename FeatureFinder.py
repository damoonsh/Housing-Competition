import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error as MSE

class FeatureSelector:
    """
        Given that the data has been processed, this module
        will try out different feature combinations and then 
        compare their results.

        Arguments:
            df: DataFrame
            estimator: the model that train will be trained and
                    evaluated on
            y_feature: independent feature that is to be predicted
    """
    def __init__(self, df, estimator, y_feature):
        self.data = df
        self.model = estimator
        self.y = y_feature
    
    def combination_generator(self, List, n_choice):
        """ 
            Generates a list of features with the choice of k out of n. 

            Arguments:
                List: list of features
                n_choice: number of choices out of all features
                
            Returns:    
                the list of different possible combinations features                
        """
        n = len(List)
        d = []
        
        if n_choice == 1 or n == n_choice: return List
        Range = n - n_choice + 1

        for i in range(0, Range):
            new_list = self.combination_generator(List[i + 1:], n_choice - 1)
            for z in new_list:
                if n_choice == len(new_list) + 1:
                    new_list.append(List[i])
                    d.append(new_list)
                    break
                if type(z) != list:
                    d.append([z, List[i]])
                else:
                    zz = z.copy()
                    zz.append(List[i])
                    d.append(zz)
        return d

    def tryOut(self, features, n, train, hand_picked=[], y_feature='SalePrice'):
        """ 
            This function tries out the different combinations and gives out the 
            performance of each of them did. 

            Arguments:
                features: list of features to tryout
                n: number of choices from all features
                train: training dataframe
                hand_picked: handpicked features
                y_feature: independent feature

            Returns:
                Dataframe containing the info about different possible feature
                combinations
        """
        # Getting the various feature combinations
        combs = self.combination_generator(features, n)
        # Setting the initial index and dict to null
        r2s = {}
        index  = 0
        # Instantiating the columns
        r2s['combination'], r2s['val1'], r2s['val2'], r2s['val3'], r2s['std'], r2s['MSE'], r2s['mean'] = [], [], [], [], [], [], []
        
        for comb in combs:
            
            if hand_picked != []:
                comb.extend(hand_picked)
            
            X = train[comb]
            y = train[y_feature]
            
            r2s['combination'].append(comb)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            
            r2 = MSE(y_pred, y_test)
            r2s['MSE'].append(r2)
            
            d = list(cross_val_score(self.model, X,  y, cv=3))
            
            r2s['val1'].append(d[0])
            r2s['val2'].append(d[1])
            r2s['val3'].append(d[2])
            r2s['mean'].append(np.mean(d))
            r2s['std'].append(np.std(d))
            
            index += 1
            
        return pd.DataFrame(r2s)

    def add_polynomial(self, df, features, max_exponent=4):
        """ 
            Adding polynomials to the features

            Arguments:
                df: dataframe
                features: list of features that should get exponent

            Returns:
                returns the new dataframe 
        """
        for feature in features:
            for exponent in range(2, max_exponent + 1):
                # Get the exponent 2, 3, 4 of the features
                title = feature + '^' + str(exponent)
                df[title] = df[feature] ** exponent       

        # Returning the new df
        return df
