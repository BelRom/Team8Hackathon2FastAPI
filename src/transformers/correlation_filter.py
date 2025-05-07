import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Удаляет признаки с:
    - низкой корреляцией с целью
    - высокой взаимной корреляцией
    """

    def __init__(self, target='y', low_thr=0.05, high_thr=0.9):
        self.target = target
        self.low_thr = low_thr
        self.high_thr = high_thr
        self.to_drop_ = []

    def fit(self, X, y=None):
        X = X.copy()

        num_cols = X.select_dtypes(include=np.number).columns.drop(self.target, errors='ignore')

        # Корреляция с целью
        corr_with_target = X[num_cols].corrwith(X[self.target]).abs()
        low_corr_feats = corr_with_target[corr_with_target < self.low_thr].index.tolist()

        # Взаимная корреляция
        corr_matrix = X[num_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
        high_corr_feats = [
            col for col in upper.columns if any(upper[col] > self.high_thr)
        ]

        self.to_drop_ = list(set(low_corr_feats + high_corr_feats))
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=self.to_drop_, errors='ignore')
        print(f"[CorrelationFilter] Удалено {len(self.to_drop_)} признаков: {self.to_drop_}")
        return X
