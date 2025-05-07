from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


class TargetMarker(BaseEstimator, TransformerMixin):
    """
    Добавляет колонку 'y' — 1, если session_id в целевых действиях.
    Также кодирует session_id через OrdinalEncoder.
    """

    def __init__(self, df_hits, targets):
        self.df_hits = df_hits
        self.targets = targets
        self.oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    def fit(self, X, y=None):
        self.target_sessions_ = (
            self.df_hits[self.df_hits['event_action'].isin(self.targets)]['session_id']
            .unique()
        )
        self.oe.fit(X[['session_id']])
        return self

    def transform(self, X):
        X = X.copy()
        X['y'] = X['session_id'].isin(self.target_sessions_).astype(int)
        X['session_id'] = self.oe.transform(X[['session_id']])
        return X
