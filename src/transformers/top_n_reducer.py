import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TopNReducer(BaseEstimator, TransformerMixin):
    """
    Оставляет только N самых популярных значений в колонке.
    Остальное помечает как 'other'.
    """

    def __init__(self, column, top_n=3, other_label='other'):
        self.column = column
        self.top_n = top_n
        self.other_label = other_label

    def fit(self, X, y=None):
        self.top_values_ = (
            X[self.column]
            .value_counts()
            .nlargest(self.top_n)
            .index
        )
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = np.where(
            X[self.column].isin(self.top_values_), X[self.column], self.other_label
        )
        return X
