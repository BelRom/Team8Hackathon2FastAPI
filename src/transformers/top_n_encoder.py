import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TopNEncoder(BaseEstimator, TransformerMixin):
    """
    Top‑N кодирование: оставляет top_n самых частых значений,
    остальное заменяет на other_label.
    """

    def __init__(self, columns, top_n=100, other_label='other'):
        self.columns = columns
        self.top_n = top_n
        self.other_label = other_label
        self.top_dict_ = {}

    def fit(self, X, y=None):
        self.top_dict_ = {}
        for col in self.columns:
            # сохраняем top_n уникальных значений
            top_values = X[col].value_counts().nlargest(self.top_n).index.tolist()
            self.top_dict_[col] = set(top_values)  # используем set для ускорения isin()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col not in X.columns:
                raise KeyError(f"Column '{col}' not found in input DataFrame.")
            tops = self.top_dict_[col]
            X[col] = np.where(X[col].isin(tops), X[col], self.other_label)
        return X
