import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier

from data_preparation import load_data, prepare_datasets
from data_transformation import make_scaler, make_transformer


SUBMISSION_PATH = '../submissions/'

# RandomForestClassifier parameters.
ESTIMATORS_RF, CRITERION_RF, DEPTH_RF, MIN_LEAF_RF, JOBS_RF = (
    500, 'gini', 20, 8, 30)

# Find these parameters with random search grid


def train_classifier(X, y):
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


def main(generate='submission'):
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
        print("feature transformation...")
        print("Projection using PCA...")
        transformer = make_transformer(X)
        X_transformed = transformer.transform(X)
        X_test_transformed = transformer.transform(X_test)
        print("Scaling...")
        scaler = make_scaler(X_transformed)
        X_transformed_and_scaled = scaler.transform(X_transformed)
        X_test_transformed_and_scaled = scaler.transform(X_test_transformed)
        print("Training classifier...")
        clf = train_classifier(X_transformed_and_scaled, y)
        print("Generating submission...")
        return generate_submission(clf,
                                   X_test_transformed_and_scaled,
                                   label_encoder,
                                   df_test['id'])


if __name__ == 'main':
    main()
