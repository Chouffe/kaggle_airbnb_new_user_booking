import pandas as pd
# import numpy as np
# from datetime import datetime
from dateutil.parser import parse
from math import floor
import itertools as it
# from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder
from utils import compose, apply_on, apply_under_threshold, load_data


# Constants for cleaning and preparing the data
CURRENT_YEAR = 2016
MISSING_VALUE = -1
AGE_COLUMN = "age"
AGE_INTERVAL_COLUMN = "age_interval"
TRANSFORM_FIRST_ACTIVE_YEAR = 'first_active_year'
TRANSFORM_FIRST_ACTIVE_MONTH = 'first_active_month'
TRANSFORM_FIRST_ACTIVE_DAY = 'first_active_day'
TRANSFORM_FIRST_ACTIVE_SEASON = 'first_active_season'
DATE_ACCOUNT_CREATED_YEAR = 'account_created_year'
DATE_ACCOUNT_CREATED_MONTH = 'account_created_month'
DATE_ACCOUNT_CREATED_DAY = 'account_created_day'
DATE_ACCOUNT_CREATED_SEASON = 'account_created_season'
CATEGORICAL_FEATURES = ['gender', 'signup_method', 'signup_flow',
                        'language', 'affiliate_channel',
                        'affiliate_provider', 'first_affiliate_tracked',
                        'signup_app', 'first_device_type', 'first_browser',
                        'account_created_season', 'first_active_season']
UNUSED_FEATURES = ['age', 'date_account_created', 'timestamp_first_active',
                   'date_first_booking', 'id']


# -------------
# Data Cleaning
# -------------


def is_missing(x):
    return x == MISSING_VALUE


def input_missing_values(df):
    return df.fillna(MISSING_VALUE).replace('-unknown-', MISSING_VALUE)


def replace_missing_values_with_mean(df, column):
    floored_avg = floor(df[df[column] != MISSING_VALUE][column].mean())
    return apply_on(df, column, lambda x: floored_avg if is_missing(x) else x)


def clean_age(df):
    """ Takes in a pandas dataframe df and cleans up age data, it adds a new
    series with column name age_interval """

    LOW_RANGE = 14
    HIGH_RANGE = 90

    def year_to_age(year):
        return CURRENT_YEAR - year

    result_df = apply_on(
        df, AGE_COLUMN,
        compose(lambda x: year_to_age(x) if x > 900 else x,
                lambda x: MISSING_VALUE if x >= HIGH_RANGE or x <= LOW_RANGE else x))
    result_df = replace_missing_values_with_mean(result_df, AGE_COLUMN)
    return result_df


def clean_data(df):
    """ Takes in a dataframe and returns a new clean dataframe.

    * Missing values are imputed
    * Age is cleaned up + age_interval is added
    * One Hot Encoding for categorical variables
    * Low frequency values are removed from first_browser
    * Low frequency values are removed from first_device_type
    * Drop unused features
    """

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
    return df_result


# -------------------
# Feature Engineering
# -------------------


def day_of_year_to_season(day_of_year):
    """ Given a day_of_year, it returns a season
    in #{spring, summer, fall, winter}"""
    assert(0 <= day_of_year <= 366)
    if 80 <= day_of_year < 172:
        return 'spring'
    elif 172 <= day_of_year < 264:
        return 'summer'
    elif 264 <= day_of_year < 355:
        return 'fall'
    else:
        return 'winter'


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


def feature_transform_age(df_train):

    def year_to_interv(year, step=4):
        intervals = range(0, 100, step)
        (idx, val) = it.dropwhile((lambda (idx, itv): itv < year),
                                  enumerate(intervals)).next()
        return idx - 1

    result_df = df_train.copy()
    result_df[AGE_INTERVAL_COLUMN] = \
        result_df[AGE_COLUMN].apply(year_to_interv)

    # Remove age_intervals with very low frequency
    result_df = apply_under_threshold(result_df,
                                      AGE_INTERVAL_COLUMN,
                                      lambda x: MISSING_VALUE,
                                      freq_threshold=.01)

    result_df = replace_missing_values_with_mean(result_df,
                                                 AGE_INTERVAL_COLUMN)
    return result_df


def feature_transform_first_active(df_train):
    df_result = apply_on(
        df_train, "timestamp_first_active", compose(parse, str))
    df_result[TRANSFORM_FIRST_ACTIVE_YEAR] = \
        df_result.timestamp_first_active.apply(lambda x: x.year)
    df_result[TRANSFORM_FIRST_ACTIVE_MONTH] = \
        df_result.timestamp_first_active.apply(lambda x: x.month)
    df_result[TRANSFORM_FIRST_ACTIVE_DAY] = \
        df_result.timestamp_first_active.apply(lambda x: x.day)
    df_result[TRANSFORM_FIRST_ACTIVE_SEASON] = \
        df_result.timestamp_first_active.apply(
            lambda x: day_of_year_to_season(x.dayofyear))
    return df_result


def feature_transform_date_account_created(df):
    df_result = apply_on(
        df, "date_account_created", compose(parse, str))
    df_result[DATE_ACCOUNT_CREATED_YEAR] = \
        df_result.date_account_created.apply(lambda x: x.year)
    df_result[DATE_ACCOUNT_CREATED_MONTH] = \
        df_result.date_account_created.apply(lambda x: x.month)
    df_result[DATE_ACCOUNT_CREATED_DAY] = \
        df_result.date_account_created.apply(lambda x: x.day)
    df_result[DATE_ACCOUNT_CREATED_SEASON] = \
        df_result.date_account_created.apply(
            lambda x: day_of_year_to_season(x.timetuple().tm_yday))
    return df_result


def feature_engineering(df):
    """ Performs all the feature engineering steps; returns a new dataframe
    * Transforms age into age_interval
    * Transforms timestamps into year/month/day/season
    * Transforms dates into year/month/day/season
    """
    df_result = feature_transform_age(df)
    df_result = feature_transform_first_active(df_result)
    df_result = feature_transform_date_account_created(df_result)
    return df_result


def prepare_datasets(df_train, df_test, target):
    """ Takes in 2 dataframes: df_train and df_test and one pandas
    series target. Returns (X, y, X_test, label_encoder) """

    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    print("...Cleaning datasets...")
    df_all_clean = clean_data(df_all)
    print("...Feature engineering...")
    df_all_clean_and_engineered = feature_engineering(df_all_clean)
    print("...One Hot Encoding...")
    df_all_clean_and_engineered = \
        one_hot_encoding(df_all_clean_and_engineered, CATEGORICAL_FEATURES)
    print("...Removing unused features...")
    df_all_clean_and_engineered = \
        df_all_clean_and_engineered.drop(UNUSED_FEATURES, axis=1)

    le = LabelEncoder()
    n = len(target)
    vals = df_all_clean_and_engineered.values

    X = vals[:n]
    X_test = vals[n:]
    y = le.fit_transform(target.values)

    return (X, y, X_test, le)


if __name__ == "__main__":
    (df_train, target, df_test) = load_data()
    (X, y, X_test, label_encoder) = prepare_datasets(df_train, df_test, target)
