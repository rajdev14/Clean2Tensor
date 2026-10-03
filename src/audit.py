import pandas as pd


def audit_data(df):

    report = {}

    report["total_rows"] = len(df)

    report["total_columns"] = len(df.columns)

    report["valid_timestamps"] = df["Timestamp"].notna().sum()

    report["invalid_timestamps"] = df["Timestamp"].isna().sum()

    report["duplicate_timestamps"] = (
        df["Timestamp"].dropna().duplicated().sum()
    )

    report["completely_empty_rows"] = (
        df.drop(columns=["Timestamp"])
        .isna()
        .all(axis=1)
        .sum()
    )

    minus_200_counts = {}

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

        count = (df[column] == -200).sum()

        if count > 0:
            minus_200_counts[column] = int(count)

    report["minus_200_counts"] = minus_200_counts

    valid_timestamps = (
        df["Timestamp"]
        .dropna()
        .sort_values()
    )

    if len(valid_timestamps) > 1:

        differences = valid_timestamps.diff().dropna()

        report["time_difference_counts"] = (
            differences.value_counts().to_dict()
        )

        gaps = differences[
            differences > pd.Timedelta(hours=1)
        ]

        report["gaps_greater_than_one_hour"] = len(gaps)

    else:

        report["time_difference_counts"] = {}

        report["gaps_greater_than_one_hour"] = 0

    return report