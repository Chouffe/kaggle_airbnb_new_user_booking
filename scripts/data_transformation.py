import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

from data_preparation import load_data, prepare_datasets
from metrics import ndcg_score


def split_data(X, y):
    """Takes in X and y and returns (X_train, X_test, y_train, y_test)"""
    return train_test_split(X, y, test_size=.25, random_state=42)


def make_transformer(X):
    pca = PCA(n_components=.94)
    pca.fit(X)
    return pca


def make_scaler(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


# XGBoost parameters.
DEPTH_XGB, ESTIMATORS_XGB, LEARNING_XGB, SUBSAMPLE_XGB, COLSAMPLE_XGB = (
    7, 60, 0.2, 0.7, 0.6)

# RandomForestClassifier parameters.
ESTIMATORS_RF, CRITERION_RF, DEPTH_RF, MIN_LEAF_RF, JOBS_RF = (
    500, 'gini', 20, 8, -1)

# Find these parameters with random search grid


def get_classifier(X, y):
    # clf = XGBClassifier(max_depth=DEPTH_XGB,
    #                     learning_rate=LEARNING_XGB,
    #                     n_estimators=ESTIMATORS_XGB,
    #                     objective='multi:softprob',
    #                     subsample=SUBSAMPLE_XGB,
    #                     colsample_bytree=COLSAMPLE_XGB)
    clf = RandomForestClassifier(
        n_estimators=ESTIMATORS_RF,
        criterion=CRITERION_RF,
        n_jobs=JOBS_RF,
        max_depth=DEPTH_RF,
        min_samples_leaf=MIN_LEAF_RF,
        bootstrap=True)
    clf.fit(X, y)
    return clf


def score_classifier(clf, X_test, y_test):
    probabilities = clf.predict_proba(X_test)
    return ndcg_score(y_test, probabilities, k=5)


def generate_submission(clf, X_test, label_encoder, ids,
                        output_filename='submission.csv'):
    y_pred = clf.predict_proba(X_test)
    # Taking the 5 classes with highest probabilities
    ids_result = []  # list of ids
    cts = []         # list of countries
    for i in range(len(ids)):
        idx = ids[i]
        ids_result += [idx] * 5
        cts += label_encoder.inverse_transform(
            np.argsort(y_pred[i])[::-1])[:5].tolist()
    # Generate submission
    sub = pd.DataFrame(np.column_stack((ids_result, cts)),
                       columns=['id', 'country'])
    sub.to_csv(output_filename, index=False)
    return sub






# transformer = make_transformer(X_tr)
# clf = get_classifier(transformer.transform(X_tr), y_tr)
# score_classifier(clf, transformer.transform(X_te), y_te)
# score_classifier(clf, transformer.transform(X_te).transpose(), y_te)

# clf = RandomForestClassifier(n_estimators=100)
# clf.fit(transformer.transform(X_tr), y_tr)
# probabilities = clf.predict_proba(transformer.transform(X_te))
# ndcg_score(y_te, probabilities, k=5)
