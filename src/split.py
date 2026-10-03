def chronological_split(df):

    df = df.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    total_rows = len(df)

    train_end = int(total_rows * 0.70)

    validation_end = int(
        total_rows * 0.85
    )

    train_df = df.iloc[
        :train_end
    ].copy()

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[
        validation_end:
    ].copy()

    return (
        train_df,
        validation_df,
        test_df
    )