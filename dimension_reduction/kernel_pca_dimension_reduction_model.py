import pandas as pd
from sklearn.decomposition import KernelPCA


class KernelPCADimensionReductionModel:
    def __init__(self, dim=10, **params):
        self.model = KernelPCA(n_components=dim)
        if len(params) > 0:
            self.model.set_params(**params)

    def fit(self, data: pd.DataFrame):
        self.model.fit(data)

    def transform(self, data: pd.DataFrame):
        return pd.DataFrame(self.model.transform(data))

    def fit_transform(self, data: pd.DataFrame):
        return pd.DataFrame(self.model.fit_transform(data))
