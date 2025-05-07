from sklearn.base import BaseEstimator, TransformerMixin

class DropTarget(BaseEstimator, TransformerMixin):
    """
    Трансформер, который удаляет колонку(и)‑цель из дата‑фрейма.
    По умолчанию удаляет 'y', но можно передать список.
    """
    def __init__(self, columns=('y',), errors: str = "ignore"):
        self.columns = columns
        self.errors = errors

    # fit просто ничего не делает и возвращает self,
    # чтобы трансформер был совместим с Pipeline
    def fit(self, X, y=None):
        return self

    # transform — главное действие
    def transform(self, X):
        # важно: работаем с копией, чтобы не менять X in‑place в пайплайне
        return X.drop(columns=list(self.columns), errors=self.errors).copy()
