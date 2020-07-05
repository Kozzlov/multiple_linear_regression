# multiple linear regression 

import numpy as np 
import matplotlib.pyplot as  plt 
import pandas as pd 

#importing the datasets 

dataset = pd.read_csv('50_Startups.csv')
#inped var vector
X = dataset.iloc[:, :-1].values
#dep var vector
Y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding the dummy variable trap 
X = X[:, 1:]

#splitting the data to the training set and the test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fitting multiple linear regression to the training set 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the test set results 
Y_pred = regressor.predict(X_test)

#Backward elimination 
import statsmodels.api as sm
X = np.append(arr =np.ones((50, 1)).astype(int), values =X, axis =1)
#Creating optical matrix of features of independent variables 
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(Y, X_opt)
regressor_OLS_result = regressor_OLS.fit()
# looking for value with the highest p value 
regressor_OLS_result.summary()

# removing column with the highest p value step 1
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(Y, X_opt)
regressor_OLS_result = regressor_OLS.fit()

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(Y, X_opt)
regressor_OLS_result = regressor_OLS.fit()
regressor_OLS_result.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(Y, X_opt)
regressor_OLS_result = regressor_OLS.fit()
regressor_OLS_result.summary()

X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(Y, X_opt)
regressor_OLS_result = regressor_OLS.fit()
regressor_OLS_result.summary()

# looking for value with the highest p value 
regressor_OLS_result.summary()
# removing column with the highest p value 


