#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:57:33 2019

@author: GalenByrd
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

housing = np.loadtxt("housing.data.txt")
housing = pd.DataFrame(housing)
housing.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
#cols = ['PerCapCrimeRate','PropResOver25000','PropNonRetailAcres','CharlesRiver','NOX','AvgRooms','Age','Distance','AccessToHwy','Tax','PTRatio','B','PerLowStatus','MedValue']
housing['CRIMLOG'] = np.log(housing['CRIM'])
housing['BLOG'] = np.log(housing['B'])
housing['ZNLOG'] = np.log(housing['ZN']+2)

summary = housing.describe()
housing.hist(figsize=(12,12))
correlations = housing.corr()

plt.scatter(housing['CRIMLOG'],housing['NOX'])
plt.xlabel('Log Crime Rate')
plt.ylabel('Nitric Oxides Concentration')

plt.scatter(housing['CRIMLOG'],housing['TAX'])
plt.xlabel('Log Crime Rate')
plt.ylabel('Property Tax Rate')


################## PART 2 ##############################
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split,RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
y = housing['CRIMLOG']
X = housing.drop(columns = ['CRIM','CRIMLOG','BLOG','ZNLOG'])
#X = housing.drop(columns = ['CRIM','CRIMLOG','BLOG','ZNLOG'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


############### LINEAR MODEL ###################
#lasso = linear_model.LassoCV(cv=5)
lasso = linear_model.Lasso()
fit = lasso.fit(X_train,y_train)

#fit = lasso.fit(X,y)
print('Test R^2: ',fit.score(X_test, y_test))
print('Train R^2: ',fit.score(X_train, y_train))

plt.bar(X.columns,lasso.coef_,width=1)
plt.title('Coefficients of Lasso Regression')
plt.xticks(rotation='vertical')

print(np.mean(cross_val_score(lasso, X_train, y_train, cv=10)))
print(np.mean(cross_val_score(lasso, X_test, y_test, cv=10)))


############### NONLINEAR MODEL ###################
regr = RandomForestRegressor()
regr.fit(X_train,y_train)
print('Test R^2: ',regr.score(X_test, y_test))
print('Train R^2: ',regr.score(X_train, y_train))

plt.bar(X.columns,list(regr.feature_importances_),width=1)
plt.title('Variable Importances in Random Forrest')
plt.xticks(rotation='vertical')

np.mean(cross_val_score(regr, X_test, y_test, cv=10))
np.mean(cross_val_score(regr, X_train, y_train, cv=10))




########################### CODEGRAVE ########################################

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
regr = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = regr, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train,y_train)
rf_random.best_params_

best_random = rf_random.best_estimator_



# Get numerical feature importances
importances = list(regr.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(housing.columns, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, housing.columns, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');



_, _, coefs = linear_model.lars_path(X, y, method='lasso', verbose=True)
xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.show()


# Alpha (regularization strength) of LASSO regression
lasso_eps = 0.0001
lasso_nalpha=20
lasso_iter=5000
# Min and max degree of polynomials features to consider
degree_min = 2
degree_max = 8

# Make a pipeline model with polynomial transformation and LASSO regression with cross-validation, run it for increasing degree of polynomial (complexity of the model)
for degree in range(degree_min,degree_max+1):
    model = make_pipeline(PolynomialFeatures(degree, interaction_only=False), linear_model.LassoCV(eps=lasso_eps,n_alphas=lasso_nalpha,max_iter=lasso_iter,normalize=True,cv=5))
    model.fit(X_train,y_train)
    test_pred = np.array(model.predict(X_test))
    RMSE=np.sqrt(np.sum(np.square(test_pred-y_test)))
    test_score = model.score(X_test,y_test)
    
    
    
    

testscores = list()
trainScores = list()
for k in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    testscores.append(lasso.fit(X_train, y_train).score(X_test, y_test))
    trainScores.append(lasso.fit(X_train, y_train).score(X_train, y_train))
 
plt.plot(list(range(10)),testscores, label = 'test')
plt.plot(list(range(10)),trainScores, label = 'train')
plt.legend()

plt.plot(list(range(10)),np.sort(testscores), label = 'test')
plt.plot(list(range(10)),np.sort(trainScores), label = 'train')
plt.legend()



lasso = linear_model.Lasso(alpha=0.1)
fit = lasso.fit(X_train,y_train)
train_score=lasso.score(X_train,y_train)
test_score=lasso.score(X_test,y_test)
coeff_used = np.sum(lasso.coef_!=0)

lasso01 = linear_model.Lasso(alpha=0.01, max_iter=10e5)
lasso01.fit(X_train,y_train)
train_score01=lasso01.score(X_train,y_train)
test_score01=lasso01.score(X_test,y_test)
coeff_used01 = np.sum(lasso01.coef_!=0)

lasso001 = linear_model.Lasso(alpha=0.001, max_iter=10e5)
lasso001.fit(X_train,y_train)
train_score001=lasso001.score(X_train,y_train)
test_score001=lasso001.score(X_test,y_test)
coeff_used001 = np.sum(lasso001.coef_!=0)

lasso00001 = linear_model.Lasso(alpha=0.0001, max_iter=10e5)
lasso00001.fit(X_train,y_train)
train_score00001=lasso00001.score(X_train,y_train)
test_score00001=lasso00001.score(X_test,y_test)
coeff_used00001 = np.sum(lasso00001.coef_!=0)


plt.plot([1,2,3,4],[train_score,train_score01,train_score001,train_score00001], label = 'train')
plt.plot([1,2,3,4],[test_score,test_score01,test_score001,test_score00001], label = 'test')
plt.legend()

predictions = fit.predict(X_test)
errors = abs(predictions - y_test)
me = np.mean(errors / y_test)
mape = 100 * me
accuracy = 100 - mape
print('Model Performance')
print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
print('Accuracy = {:0.2f}%.'.format(accuracy))