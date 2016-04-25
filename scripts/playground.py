import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.cross_validation import train_test_split

from data_preparation import load_data, prepare_datasets
from data_transformation import make_transformer, make_scaler
from metrics import ndcg_scorer


(df_train, target, df_test) = load_data()
(X, y, X_test, label_encoder) = prepare_datasets(df_train, df_test, target)
(X_train, X_te, y_train, y_te) = train_test_split(
    X, y, test_size=.25, random_state=41)
transformer = make_transformer(X_train)


def main():
    parameters = {'n_estimators': [3, 10, 50, 300],
                  'criterion': ["gini", "entropy"],
                  'max_features': ["sqrt", "log2", 3],
                  'max_depth': [3, 4, 5, 6],
                  'min_samples_leaf': [1, 2, 3],
                  'min_samples_split': [2, 3, 4, 5],
                  }
    rf = RandomForestClassifier(n_jobs=-1)
    clf = RandomizedSearchCV(rf, parameters, pre_dispatch=1, cv=10, verbose=10, scoring=ndcg_scorer)
    clf.fit(X_train, y_train)
    return clf


if __name__ == 'main':
    main()
