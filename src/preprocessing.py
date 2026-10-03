import pandas as pd

from .features import FEATURE_COLUMNS


class MedianImputer:

    def __init__(self):
        self.medians = None

    def fit(self, df):

        self.medians = (
            df[FEATURE_COLUMNS]
            .median()
        )

        return self

    def transform(self, df):

        if self.medians is None:
            raise RuntimeError(
                "Imputer must be fitted before transform."
            )

        result = df.copy()

        result[FEATURE_COLUMNS] = (
            result[FEATURE_COLUMNS]
            .fillna(self.medians)
        )

        return result

    def fit_transform(self, df):

        self.fit(df)

        return self.transform(df)


class StandardScaler:

    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, df):

        self.means = (
            df[FEATURE_COLUMNS]
            .mean()
        )

        self.stds = (
            df[FEATURE_COLUMNS]
            .std()
        )

        return self

    def transform(self, df):

        if self.means is None or self.stds is None:
            raise RuntimeError(
                "Scaler must be fitted before transform."
            )

        result = df.copy()

        result[FEATURE_COLUMNS] = (
            result[FEATURE_COLUMNS]
            - self.means
        ) / self.stds

        return result

    def fit_transform(self, df):

        self.fit(df)

        return self.transform(df)