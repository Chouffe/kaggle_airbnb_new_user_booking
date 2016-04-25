import pandas as pd
# import numpy as np
from math import floor
import itertools as it
# from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder
from utils import compose, apply_on, apply_under_threshold


def load_data():
    """Loads data in memory. Returns two dataframes and one series
    Returns (df_train, target, df_test).
    """
    df_raw_train_data = pd.read_csv("../data/train_users_2.csv")
    df_test = pd.read_csv("../data/test_users.csv")
    target = df_raw_train_data['country_destination']
    df_train = df_raw_train_data.drop(['country_destination'], axis=1)
    return (df_train, target, df_test)


MISSING_VALUE = -1


def is_missing(x):
    return x == MISSING_VALUE


def input_missing_values(df):
    return df.fillna(MISSING_VALUE).replace('-unknown-', MISSING_VALUE)

# TODO: clean dates + feature engineering on dates


def clean_age(df):
    """ Takes in a pandas dataframe df and cleans up age data, it adds a new
    series with column name age_interval """

    AGE_COLUMN = "age"
    AGE_INTERVAL_COLUMN = "age_interval"
    CURRENT_YEAR = 2016
    LOW_RANGE = 14
    HIGH_RANGE = 90

    def year_to_age(year):
        return CURRENT_YEAR - year

    def year_to_interv(year, step=4):
        intervals = range(0, 100, step)
        (idx, val) = it.dropwhile((lambda (idx, itv): itv < year),
                                  enumerate(intervals)).next()
        return idx - 1

    def replace_missing_values_with_mean(df, column):
        floored_avg = floor(df[df[column] != MISSING_VALUE][column].mean())
        return apply_on(result_df,
                        column,
                        lambda x: floored_avg if is_missing(x) else x)

    result_df = apply_on(
        df, AGE_COLUMN,
        compose(lambda x: year_to_age(x) if x > 900 else x,
                lambda x: MISSING_VALUE if x >= HIGH_RANGE or x <= LOW_RANGE else x))

    result_df = replace_missing_values_with_mean(result_df, AGE_COLUMN)
    result_df[AGE_INTERVAL_COLUMN] = result_df[AGE_COLUMN].apply(year_to_interv)

    # Remove age_intervals with very low frequency
    result_df = apply_under_threshold(result_df,
                                      AGE_INTERVAL_COLUMN,
                                      lambda x: MISSING_VALUE,
                                      freq_threshold=.01)

    result_df = replace_missing_values_with_mean(result_df,
                                                 AGE_INTERVAL_COLUMN)

    return result_df


def one_hot_encoding(df, columns):
    """ Takes in a dataframe df and an iterable for column names.
    Returns a dataframe with the dummy features added.
    """
    df_result = df.copy()
    for column in columns:
        df_dummy = pd.get_dummies(df[column], prefix=column)
        df_result = df_result.drop([column], axis=1)
        df_result = pd.concat((df_result, df_dummy), axis=1)
    return df_result


def clean_data(df):
    """ Takes in a dataframe and returns a new clean dataframe.


    * Missing values are inputed
    * Age is cleaned up + age_interval is added
    * One Hot Encoding for categorical variables
    * Low frequency values are removed from first_browser
    * Low frequency values are removed from first_device_type
    * Drop unused features
    """

    CATEGORICAL_FEATURES = ['gender', 'signup_method', 'signup_flow',
                            'language', 'affiliate_channel',
                            'affiliate_provider', 'first_affiliate_tracked',
                            'signup_app', 'first_device_type', 'first_browser']
    UNUSED_FEATURES = ['date_first_booking', 'id',
                       'date_account_created', 'timestamp_first_active']

    # Compose cleaning steps here
    df_result = input_missing_values(df)
    df_result = clean_age(df_result)
    df_result = apply_under_threshold(df_result,
                                      'first_browser',
                                      lambda x: 'OTHER',
                                      freq_threshold=.01)
    df_result = apply_under_threshold(df_result,
                                      'first_device_type',
                                      lambda x: 'Other/Unknown',
                                      freq_threshold=.005)
    df_result = one_hot_encoding(df_result, CATEGORICAL_FEATURES)
    df_result = df_result.drop(UNUSED_FEATURES, axis=1)
    return df_result


def prepare_datasets(df_train, df_test, target):
    """ Takes in 2 dataframes: df_train and df_test and one pandas
    series target. Returns (X, y, X_test, label_encoder) """

    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df_all_clean = clean_data(df_all)

    le = LabelEncoder()
    n = len(target)
    vals = df_all_clean.values

    X = vals[:n]
    X_test = vals[n:]
    y = le.fit_transform(target.values)

    return (X, y, X_test, le)


if __name__ == "__main__":
    (df_train, target, df_test) = load_data()
    (X, y, X_test, label_encoder) = prepare_datasets(df_train, df_test, target)
