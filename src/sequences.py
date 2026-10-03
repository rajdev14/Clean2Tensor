import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, TARGET_COLUMN


WINDOW_SIZE = 24


def create_sequences(df):

    df = df.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    X = []
    y = []
    timestamps = []

    for end_index in range(
        WINDOW_SIZE,
        len(df)
    ):

        start_index = (
            end_index - WINDOW_SIZE
        )

        window = df.iloc[
            start_index:end_index
        ]

        target = df.iloc[
            end_index
        ][TARGET_COLUMN]

        if pd.isna(target):
            continue

        X.append(
            window[FEATURE_COLUMNS]
            .to_numpy(dtype=np.float32)
        )

        y.append(
            np.float32(target)
        )

        timestamps.append(
            df.iloc[end_index]["Timestamp"]
        )

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.float32
    )

    return X, y, timestamps