from sklearn.base import BaseEstimator, TransformerMixin


class DropColumns(BaseEstimator, TransformerMixin):
    """Удаляет заданные столбцы."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns)
