# import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier

from sklearn.grid_search import RandomizedSearchCV # , GridSearchCV
from sklearn.cross_validation import train_test_split

from data_preparation import load_data, prepare_datasets
from data_transformation import feature_transformation
from metrics import ndcg_scorer

from evaluation import score_classifier, display_learning_curves


print("loading data in memory...")
(df_train, target, df_test) = load_data()

print("preparing datasets...")
(X, y, X_test, label_encoder) = prepare_datasets(df_train, df_test, target)

print("Transforming datasets...")
(X_transformed, scaler, transformer) = feature_transformation(X)
X_test_scaled = scaler.transform(X_test)
X_test_transformed = transformer.transform(X_test_scaled)

print("splitting data ")
(X_train, X_te, y_train, y_te) = train_test_split(
    X_transformed, y, test_size=.25, random_state=41)


def main():
    parameters = {'n_estimators': [3, 10, 50, 300],
                  'criterion': ["gini", "entropy"],
                  'max_features': ["sqrt", "log2", 3],
                  'max_depth': [3, 4, 5, 6],
                  'min_samples_leaf': [1, 2, 3],
                  'min_samples_split': [2, 3, 4, 5],
                  }
    rf = RandomForestClassifier(n_jobs=-1)
    clf = RandomizedSearchCV(
        rf, parameters, pre_dispatch=1, cv=10, verbose=10, scoring=ndcg_scorer)
    clf.fit(X_train, y_train)
    return clf

# ESTIMATORS_RF, CRITERION_RF, DEPTH_RF, MIN_LEAF_RF, JOBS_RF = (
#     500, 'gini', 20, 8, -1)


def main2():
    print("Training a classifier...")
    clf = RandomForestClassifier(criterion='gini',
                                 n_jobs=-1,
                                 n_estimators=500,
                                 max_depth=20,
                                 min_samples_leaf=8)
    clf.fit(X_train, y_train)
    print("evaluation...")
    score = score_classifier(clf, X_te, y_te)
    print("NDCG@5 score: {}".format(score))


def evaluate_classifiers(X, y):
    classifiers = [
        # ("Decision Tree Classifier", DecisionTreeClassifier()),
        #            ("Gaussian Naive Bayes", GaussianNB()),
                   ("Random Forest Classifier",
                    RandomForestClassifier(n_jobs=-1, n_estimators=500)),
                   ("XGBOOST Classifier",
                    XGBClassifier(n_estimators=500,
                                  objective='multi:softprob')),
                   ("SVM Classifier", SVC(probability=True))
                   ]

    for (clf_name, clf) in classifiers:
        print(clf_name)
        display_learning_curves(clf, X, y, clf_name)
    plt.show()


if __name__ == '__main__':
    evaluate_classifiers(X_transformed, y)
