import pandas as pd


def generate_statistics(df):

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    statistics = []

    for column in numeric_columns:

        valid_values = df[column].dropna()

        statistics.append({
            "feature": column,
            "valid_count": len(valid_values),
            "missing_count": int(
                df[column].isna().sum()
            ),
            "missing_percentage": round(
                df[column].isna().mean() * 100,
                2
            ),
            "minimum": valid_values.min(),
            "maximum": valid_values.max(),
            "mean": valid_values.mean(),
            "median": valid_values.median(),
            "std": valid_values.std()
        })

    return pd.DataFrame(statistics)