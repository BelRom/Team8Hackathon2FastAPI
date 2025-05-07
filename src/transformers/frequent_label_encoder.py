from sklearn.base import BaseEstimator, TransformerMixin


class FrequentLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Кодирует 'geo_country_frequent' и 'geo_city_frequent',
    заменяя неизвестные значения (и 'Other') на -1.
    """

    def __init__(self):
        self.encoders_ = {}
        self.known_values_ = {}

    def fit(self, X, y=None):
        for col in ['geo_country_frequent', 'geo_city_frequent']:
            # Заменяем 'Other' на -1, сохраняем как строку
            values = X[col].replace('Other', -1).astype(str)
            self.known_values_[col] = sorted(values.unique())
            self.encoders_[col] = {v: i for i, v in enumerate(self.known_values_[col])}
        return self

    def transform(self, X):
        X = X.copy()
        for col in ['geo_country_frequent', 'geo_city_frequent']:
            values = X[col].replace('Other', -1).astype(str)
            encoder = self.encoders_[col]
            X[col] = values.apply(lambda v: encoder.get(v, -1)).astype(int)
        return X
