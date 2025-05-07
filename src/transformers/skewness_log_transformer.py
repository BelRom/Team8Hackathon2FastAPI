import numpy as np
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin


class SkewnessLogTransformer(BaseEstimator, TransformerMixin):
    """
    Применяет логарифм log1p к признакам с асимметрией > threshold.
    Создаёт новые колонки с суффиксом '_log'.
    """

    def __init__(self, numeric_features, threshold=1.0):
        self.numeric_features = numeric_features
        self.threshold = threshold
        self.skewed_cols_ = []

    def fit(self, X, y=None):
        skewness = X[self.numeric_features].apply(lambda x: skew(x.dropna()))
        self.skewed_cols_ = skewness[skewness > self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.skewed_cols_:
            X[f"{col}_log"] = np.log1p(X[col])
        return X
