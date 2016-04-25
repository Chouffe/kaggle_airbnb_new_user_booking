import numpy as np


def compose(f, g):
    """Composes f and g"""
    def h(x):
        return f(g(x))
    return h


def apply_on(df, column, f):
    """Retrun a new dataframe after applying f on column"""
    result = df.copy()
    result.loc[:, column] = df[column].apply(f)
    return result


def apply_under_threshold(df, column, f, freq_threshold=.001):
    """applies F on values in a column if under a freq_threshold"""
    frequencies = df[column].value_counts(normalize=True)
    return apply_on(df, column,
                    lambda x: f(x) if frequencies[x] < freq_threshold else x)


def largest_k_indexes(coll, k=5):
    """Returns the k index mapping to the k largest values in coll"""
    return np.argsort(coll)[::-1][:k].tolist()
