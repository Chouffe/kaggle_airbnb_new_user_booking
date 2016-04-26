import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier

from data_preparation import load_data, prepare_datasets
from data_transformation import feature_transformation


# Path for the submission.csv file
SUBMISSION_PATH = '../submissions/'

# RandomForestClassifier parameters.
ESTIMATORS_RF = 500
CRITERION_RF = 'gini'
JOBS_RF = -1
DEPTH_RF = 20
MIN_LEAF_RF = 8
MIN_SPLIT_RF = 2
MAX_FEATURES_RF = 'sqrt'

# XGBOOST Classifier parameters
ESTIMATORS_XG = 500
OBJECTIVE_XG = 'multi:softprob'
DEPTH_XG = 6
LEARNING_RATE_XG = 0.3
SUBSAMPLE_XG = 0.5
COLSAMPLE_BYTREE_XG = 0.5


def train_classifier(X, y, clf_name='xgb'):
    if clf_name == 'xgb':
        clf = XGBClassifier(
            n_estimators=ESTIMATORS_XG,
            objective=OBJECTIVE_XG,
            max_depth=DEPTH_XG,
            learning_rate=LEARNING_RATE_XG,
            subsample=SUBSAMPLE_XG,
            colsample_bytree=COLSAMPLE_BYTREE_XG,
            seed=0,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=ESTIMATORS_RF,
            criterion=CRITERION_RF,
            n_jobs=JOBS_RF,
            max_depth=DEPTH_RF,
            min_samples_leaf=MIN_LEAF_RF,
            min_samples_split=MIN_SPLIT_RF,
            max_features=MAX_FEATURES_RF,
            bootstrap=True,
        )
    clf.fit(X, y)
    return clf


def df_submission(ids, f):
    ids_result = []  # list of ids
    cts = []         # list of countries
    for i in range(len(ids)):
        idx = ids[i]
        ids_result += [idx] * 5
        cts += f(i)
    return pd.DataFrame(np.column_stack((ids_result, cts)),
                        columns=['id', 'country'])


def generate_benchmark1_submission(df_train, target, ids):
    df_sub = df_submission(ids, lambda i: ['NDF'])
    df_sub.to_csv(SUBMISSION_PATH + 'benchmark1.csv', index=False)
    return df_sub


def generate_benchmark2_submission(df_train, target, ids):
    prediction = target.value_counts().index.tolist()[:5]
    df_sub = df_submission(ids, lambda i: prediction)
    df_sub.to_csv(SUBMISSION_PATH + 'benchmark2.csv', index=False)
    return df_sub


def generate_submission(clf, X_test, label_encoder, ids,
                        output_filename=SUBMISSION_PATH + 'submission.csv'):
    y_pred = clf.predict_proba(X_test)
    df_sub = df_submission(
        ids, lambda i: label_encoder.inverse_transform(
            np.argsort(y_pred[i])[::-1])[:5].tolist())
    df_sub.to_csv(output_filename, index=False)
    return df_sub


def main(generate='xgb', scaling=False):
    """ Runs the different processing steps to generate a
    solution in csv format"""

    print("loading data in memory...")
    (df_train, target, df_test) = load_data()

    print("preparing datasets...")
    (X, y, X_test, label_encoder) = prepare_datasets(df_train, df_test, target)

    if generate == 'benchmark1':
        print("Generating submission...")
        return generate_benchmark1_submission(df_train,
                                              target,
                                              df_test['id'])
    elif generate == 'benchmark2':
        print("Generating submission...")
        return generate_benchmark2_submission(df_train,
                                              target,
                                              df_test['id'])
    else:
        if scaling:
            print("Transforming datasets...")
            (X, scaler, transformer) = feature_transformation(X)
            X_test = transformer.transform(scaler.transform(X_test))

        print("Training classifier...")
        clf = train_classifier(X, y)

        print("Generating submission...")
        return generate_submission(
            clf, X_test, label_encoder, df_test['id'], clf_name=generate)


if __name__ == '__main__':
    main(generate='xgb')
