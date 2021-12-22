import pandas as pd

class BaseDimensionReductionModel:
    def fit(self, data: pd.DataFrame):
        pass

    def transform(self, data: pd.DataFrame):
        pass

    def fit_transform(self, data: pd.DataFrame):
        pass