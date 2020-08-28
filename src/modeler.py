import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

class CvdModeler(object):
    """
    sklearn model to use for modeling a binary cvd risk
    """

    def __init__(self, model):
        self.model = model

    def print_model_metrics(self, X_train, X_test, y_train, y_test):
        """
        Print model performance metrics
        Args:
            X_train: ndarray - 2D
            X_test: ndarray - 2D
            y_train: ndarray - 1D
            y_test: ndarray - 1D
        Returns:
            Nothing, just prints
        """
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        name = self.model.__class__.__name__.replace('Classifier','')
        print('*'*30)
        print("{} Accuracy (test):".format(name), accuracy_score(y_test, y_pred))
        print("{} Precision (test):".format(name), precision_score(y_test, y_pred))
        print("{} Recall (test):".format(name), recall_score(y_test, y_pred))

    def plot_feature_importance(self, X, col_names):
        """
        Plots feature importance (for random forest and gradient boost models)
        Args:
            X: ndarray - 2D
            col_names(list): column names of X
        Returns:
            Feature importance plot
        """
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        name = self.model.__class__.__name__.replace('Classifier','')
        plt.bar(range(X.shape[1]), importances[indices], color="b")
        plt.title("{} Feature Importances".format(name))
        plt.xlabel("Feature")
        plt.ylabel("Feature importance")
        plt.xticks(range(X.shape[1]), col_names[indices], rotation=45, fontsize=12, ha='right')
        plt.xlim([-1, X.shape[1]])
        plt.tight_layout()

def load_split_data(select=0):
    """ 
        Load data in
    Args:
        select(int): option to control whether selective features will be used 
    Returns:
        Train_test datasets for X and y, as well as a list for column names
    """
    df_cvd = pd.read_csv('data/final_df')
    if select:
        feature_choice = ['age', 'systolic', 'diastolic', 'BMI, 'cholestrol_1', 'cholestrol_2', 'cholestrol_3', 'gluc_1', 'gluc_2', 'gluc_3', 'cvd']
        df_cvd = df_cvd[feature_choice]

    y = df_cvd.pop('cvd').values
    X = df_cvd.values
    col_names = df_cvd.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    return (X_train, X_test, y_train, y_test), col_names