import numpy as np
from datetime import datetime
import pickle

from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV # , GridSearchCV
from sklearn.cross_validation import train_test_split
from metrics import ndcg_scorer
from evaluation import score_classifier
from data_preparation import load_data, prepare_datasets


HYPER_TUNING_PATH = 'hyperparameter_tuning/'
HYPER_TUNING_SCORE = 'score.txt'

PARAMETERS_RF = {'n_estimators': [50],
                 'criterion': ["gini", "entropy"],
                 'max_features': ["sqrt", "log2"] + np.linspace(2, 50, 10, dtype='int').tolist(),
                 'max_depth': np.linspace(5, 50, 10, dtype='int').tolist(),
                 'min_samples_leaf': np.linspace(1, 20, 5, dtype='int').tolist(),
                 'min_samples_split': np.linspace(2, 10, 4, dtype='int').tolist(),
                 }

PARAMETERS_XG = {'n_estimators': [50],
                 'objective': ['multi:softprob'],
                 'max_depth': np.linspace(2, 20, 5, dtype='int').tolist(),
                 'learning_rate': np.linspace(0.05, 0.5, 10).tolist(),
                 'subsample': np.linspace(0.2, 0.8, 10).tolist(),
                 'colsample_bytree': np.linspace(0.2, 0.8, 10).tolist()
                 }


def hyperparameter_tuning(clf, parameters, X, y):
    rs = RandomizedSearchCV(
        clf, parameters, pre_dispatch=1,
        cv=10, verbose=10,
        n_jobs=-1, scoring=ndcg_scorer,
        n_iter=20)  # todo: change to 100
    rs.fit(X, y)
    return rs


def persist_clf(clf, filename):
    with open(HYPER_TUNING_PATH + filename, 'wb') as fid:
        pickle.dump(clf, fid)


def persist_score(clf_name, clf, score):
    with open(HYPER_TUNING_PATH + HYPER_TUNING_SCORE, 'a') as fid:
        fid.write('{} - {}: {} -> {}\n\n'.format(str(datetime.now()),
                                         clf_name,
                                         clf.get_params,
                                         score))


def datasets(scaling=False):
    """ Prepares the datasets and returns (X_train, X_test, y_train, y_test)"""

    print("loading data in memory...")
    (df_train, target, df_test) = load_data()

    print("preparing datasets...")
    (X, y, _, _) = prepare_datasets(df_train, df_test, target)

    if scaling:
        print("Transforming datasets...")
        (X, _, _) = feature_transformation(X)

    return train_test_split(X, y, test_size=.30, random_state=42)


def main(clf_name='xgb'):

    (X_train, X_test, y_train, y_test) = datasets(scaling=False)

    print("Training classifier..." + clf_name)
    if clf_name == 'xgb':
        clf = XGBClassifier()
        parameters = PARAMETERS_XG
    else:
        clf = RandomForestClassifier(n_jobs=-1)
        parameters = PARAMETERS_RF

    # Hyperparameter tuning
    clf = hyperparameter_tuning(clf, parameters, X_train, y_train)

    print('Persisting classifier...' + clf_name)
    persist_clf(clf, clf_name + '-' + str(datetime.now()))

    # Score on test data
    best_clf = clf.best_estimator_
    score = score_classifier(best_clf, X_test, y_test)
    print("NDCG@5 score: {}".format(score))
    print("Persisting score...")
    persist_score(clf_name, best_clf, score)


if __name__ ==  '__main__':
    main()
