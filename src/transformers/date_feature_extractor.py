import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Извлекает год/месяц/день/день_недели/is_weekend/час визита
    и удаляет исходные visit_date / visit_time.
    """

    def __init__(self,
                 date_col='visit_date',
                 time_col='visit_time',
                 date_fmt='%Y-%m-%d',
                 time_fmt='%H:%M:%S'):
        self.date_col = date_col
        self.time_col = time_col
        self.date_fmt = date_fmt
        self.time_fmt = time_fmt

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        date = pd.to_datetime(X[self.date_col], format=self.date_fmt)

        X['year'] = date.dt.year
        X['month'] = date.dt.month
        X['day'] = date.dt.day
        X['dayofweek'] = date.dt.day_of_week
        X['is_weekend'] = X['dayofweek'].isin([5, 6]).astype(int)

        X['visit_hour'] = pd.to_datetime(X[self.time_col],
                                         format=self.time_fmt).dt.hour

        X = X.drop(columns=[self.date_col, self.time_col])
        return X
