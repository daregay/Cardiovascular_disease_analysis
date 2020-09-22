# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import math
# models
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# train-test-split
from sklearn.model_selection import train_test_split, KFold
# hyperparameter tuning parameter search methodology
from sklearn.model_selection import RandomizedSearchCV
#  metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification
# other tools
from statsmodels.tools import add_constant
from scipy.stats import uniform
from statsmodels.discrete.discrete_model import Logit
from sklearn.preprocessing import StandardScaler

def split_train_holdout(df, target, test_size=0.2, random_state=33):
    '''takes dataframe specifies the feature variables and the target variable, 
    divides the data into the training and holdout data using given test_size, and random_state
    '''
    y= df[target].values
    del df[target]
    X= df.values
    scaler= StandardScaler()
    scaled_x= scaler.fit_transform(X)
    X_train, X_holdout, y_train, y_holdout = train_test_split(scaled_x, y, 
                                                          shuffle=True,
                                                          test_size=test_size, 
                                                          random_state=random_state)
    return X_train, X_holdout, y_train, y_holdout 


def gradient_boosting(X_train, y_train):
    '''finds the best parameters for Gradient Boosting classifier using randomed search cv'''
    # declare the classifier
    gclassifier = GradientBoostingClassifier()
    # parameters to investigate
    num_estimators = [8, 15, 50, 100]
    criterion = ['friedman_mse', 'mse', 'mae']
    max_features = ['auto', 'sqrt', "log2"]
    max_depth = [5, 10, None]
    min_samples_split = [2, 10, 30]
    min_samples_leaf = [1, 3]
    # dictionary containing the parameters
    param_grid = {'n_estimators': num_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'criterion': criterion}
    # hyperparameter tuning parameter search methodology
    g_random_search = RandomizedSearchCV(gclassifier, 
                                   param_grid, 
                                   scoring='recall',
                                   cv=3,
                                   n_iter=200,
                                   verbose=1,
                                   return_train_score=True,
                                   n_jobs=-1)
    g_random_search.fit(X_train, y_train)
    # print the bestparameters and the best recall score
    gradientboosting_bestparams = g_random_search.best_params_
    gradientboosting_bestscore =  g_random_search.best_score_
    return gradientboosting_bestparams , gradientboosting_bestscore

def random_forest(X_train, y_train):
    '''finds the best parameters for Random Forest classifier using randomed search cv'''
    # declare the classifier
    classifier = RandomForestClassifier()
    # parameters to investigate
    num_estimators = [90,150,300,1000]
    criterion = ["gini", 'entropy']
    max_features = ['auto', 'sqrt', "log2"]
    max_depth = [20, 10, None]
    min_samples_split = [2, 3, 10]
    min_samples_leaf = [3, 5,7]
    # dictionary containing the parameters
    param_grid = {'n_estimators': num_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'criterion': criterion}
    # hyperparameter tuning parameter search methodology
    random_search = RandomizedSearchCV(classifier, 
                                   param_grid, 
                                   scoring='recall',
                                   cv=3,
                                   n_iter=200,
                                   verbose=1,
                                   return_train_score=True,
                                   n_jobs=-1)
    random_search.fit(X_train, y_train)
    # print the bestparameters and the best recall score
    randomforest_randomsearch_bestparams = random_search.best_params_
    randomforest_randomsearch_bestscore =  random_search.best_score_
    return randomforest_randomsearch_bestparams, randomforest_randomsearch_bestscore 


def logistic_regression(X_train, y_train, solver=['liblinear']):
    '''finds the best parameters for Logistic Regresssion classifier using randomed search cv
    for a given solver as a list'''
    # declare the classifier
    logistic = LogisticRegression()
    # parameters to investigate
    if solver == ['lbfgs'] or solver == ['newton-cg'] or solver== ['sag']:
        penalty = ['l2', 'none']
    elif solver == ['liblinear']:
        penalty = ['l1', 'l2']
    elif solver == ['sage']:
        penalty = ['elasticnet']
    distributions = dict(C=uniform(loc=0, scale=4),
                    penalty=penalty,
                    max_iter= [100, 200, 500, 1000],
                    solver= solver)
    # hyperparameter tuning parameter search methodology
    desired_iterations = 100
    search = RandomizedSearchCV(logistic, 
                                   distributions, 
                                   scoring='recall',
                                   cv=3,
                                   n_iter=desired_iterations,
                                   verbose=1,
                                   return_train_score=True,
                                   n_jobs=-1, 
                                   random_state=0)
    log_search = search.fit(X_train, y_train)
    logistic_randomsearch_bestparams = log_search.best_params_
    logistic_randomsearch_bestscore =  log_search.best_score_
    return logistic_randomsearch_bestparams, logistic_randomsearch_bestscore


def finalmodel(X_train, y_train, X_holdout, y_holdout,Classifier, **kwargs):
    ''' given classifier and the best parameters (**kwargs) for that classifier, it returns the
    predict, score, precision, and recall'''
    final_model= Classifier(**kwargs)
    final_model.fit(X_train, y_train)
    y_predict= final_model.predict(X_holdout)
    score=final_model.score(X_holdout, y_holdout)
    precision=precision_score(y_holdout, y_predict)
    recall=recall_score(y_holdout, y_predict)
    confusion_matrix=confusion_matrix(y_holdout, y_predict)
    dic={"score": score, "precision": precision, 'recall':recall, 'confusion_matrix': confusion_matrix}
    return (f'for {Classifier} the final model with optimized hyperparameters results in {dic}')

if __name__ == "__main__":
    # import the dataframe and split the data
    model_df= pd.read_csv('data/final_df.csv')
    # model_df= df2.copy()
    X_train, X_holdout, y_train, y_holdout= split_train_holdout(model_df, 'cardio')
    # run randomsearchcv for gradientboosting and use the result bestparameters to create the scores for the final model
    gradientboosting_randomsearch_bestparams,gradientboosting_randomsearch_bestscore = gradient_boosting(X_train, y_train)
    gradientboosting_finalmodel= finalmodel(X_train, y_train, X_holdout, y_holdout,GradientBoostingClassifier,gradientboosting_randomsearch_bestparams)
    # run randomsearchcv for randomforest and use the result bestparameters to create the scores for the final model
    randomforest_randomsearch_bestparams , randomforest_randomsearch_bestscore = random_forest(X_train, y_train)
    randomforest_finalmodel= finalmodel(X_train, y_train, X_holdout, y_holdout,RandomForestClassifier,randomforest_randomsearch_bestparams)
    # scale data then run randomsearchcv for logistic regression and use the result bestparameters to create the scores for the final model
    logistic_randomsearch_bestparams, logistic_randomsearch_bestscore= logistic_regression(X_trainn, y_trainn)
    logisticregression_finalmodel=finalmodel(X_trainn, y_trainn, X_test, y_test,LogisticRegression, logistic_randomsearch_bestparams)