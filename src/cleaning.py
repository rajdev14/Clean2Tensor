import pandas as pd


TARGET_COLUMN = "NO2(GT)"


def clean_data(df):

    cleaned = df.copy()

    initial_rows = len(cleaned)

    # Remove rows without a valid timestamp
    cleaned = cleaned.dropna(
        subset=["Timestamp"]
    ).copy()

    rows_after_timestamp = len(cleaned)

    # Convert -200 values to missing values
    numeric_columns = cleaned.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        cleaned.loc[
            cleaned[column] == -200,
            column
        ] = pd.NA

    # Remove duplicate timestamps
    duplicate_count = cleaned[
        "Timestamp"
    ].duplicated().sum()

    cleaned = cleaned.drop_duplicates(
        subset=["Timestamp"]
    ).copy()

    # Sort chronologically
    cleaned = cleaned.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    report = {
        "initial_rows": initial_rows,
        "rows_after_timestamp_validation": rows_after_timestamp,
        "invalid_timestamp_rows_removed":
            initial_rows - rows_after_timestamp,
        "duplicate_timestamps_removed":
            int(duplicate_count),
        "final_rows": len(cleaned),
        "missing_target_values":
            int(cleaned[TARGET_COLUMN].isna().sum())
    }

    return cleaned, report