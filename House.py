from utils import *


""" 
    Setting up the initial values 
"""
# The value that we are to predict
y_feature = 'SalePrice'

# Pass no parameters since it will fetch the training data
train = load_house_data()
test = load_house_data("test.csv")