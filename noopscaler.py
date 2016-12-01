class NoOpScaler:
    @staticmethod
    def fit_transform(X):
        return X

    @staticmethod
    def transform(X):
        return X

    @staticmethod
    def inverse_transform(X):
        return X
