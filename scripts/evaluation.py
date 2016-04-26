import numpy as np
from metrics import ndcg_score, ndcg_scorer
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
from utils import save_plot, plot_learning_curve

sns.set_style('whitegrid')


EVALUATION_PATH = 'evaluation'


def score_classifier(clf, X_test, y_test):
    """ Given a classifier clf, and a Test set (X_test, y_test),
    It scores it by returning its NDCG@5 score"""
    probabilities = clf.predict_proba(X_test)
    return ndcg_score(y_test, probabilities, k=5)


def display_learning_curves(clf, X, y, title="Learning Curves"):
    return plot_learning_curve(clf, title, X, y,
                               n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5),
                               scoring=ndcg_scorer)


def get_learning_curves(clf, X, y):
    # train_sizes, train_scores, valid_scores = learning_curve(
    #     clf, X, y, train_sizes=[.20, .30, .40, .50, .60, .70, .80, .90, .95],
    #     cv=5, scoring=ndcg_scorer, verbose=10)
    train_sizes, train_scores, valid_scores = learning_curve(
        clf, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1, cv=5,
        scoring=ndcg_scorer, verbose=10)
    # train_sizes, train_scores, valid_scores = learning_curve(
    #     clf, X, y, train_sizes=[.20, .70, .95], cv=5)
    return (train_sizes, train_scores, valid_scores)


# plot_learning_curve(rf, "RandomForestClassifier", X_transformed, y, n_jobs=-1,
#                     train_sizes=np.linspace(0.1, 1.0, 10))

# display(train_sizes, train_scores.mean(axis=1), valid_scores.mean(axis=1))
