from sklearn.base import BaseEstimator, TransformerMixin

class FillUnknown(BaseEstimator, TransformerMixin):
    """Заполняет NaN в выбранных категориальных колонках строкой 'unknown'."""
    def __init__(self, cols, fill_value='unknown'):
        self.cols = cols
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cols] = X[self.cols].fillna(self.fill_value)
        return X