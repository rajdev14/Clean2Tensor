FEATURE_COLUMNS = [
    "PT08.S1(CO)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH"
]

TARGET_COLUMN = "NO2(GT)"


def select_features(df):

    required_columns = (
        FEATURE_COLUMNS +
        [TARGET_COLUMN, "Timestamp"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    selected_df = df[
        required_columns
    ].copy()

    return selected_df