import pandas as pd


TARGET_COLUMN = "NO2(GT)"


def analyze_missing_values(df):

    report = {}

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    input_columns = [
        column
        for column in numeric_columns
        if column != TARGET_COLUMN
    ]

    missing_summary = []

    for column in input_columns:

        missing_count = int(
            df[column].isna().sum()
        )

        missing_percentage = (
            missing_count / len(df)
        ) * 100

        missing_summary.append({
            "feature": column,
            "missing_count": missing_count,
            "missing_percentage":
                round(missing_percentage, 2)
        })

    report["feature_missingness"] = (
        pd.DataFrame(missing_summary)
    )

    input_missing_mask = df[
        input_columns
    ].isna()

    report["rows_with_any_missing_input"] = int(
        input_missing_mask.any(axis=1).sum()
    )

    report["rows_with_all_inputs_available"] = int(
        (~input_missing_mask.any(axis=1)).sum()
    )

    report["rows_with_multiple_missing_inputs"] = int(
        (input_missing_mask.sum(axis=1) > 1).sum()
    )

    report["missing_target_count"] = int(
        df[TARGET_COLUMN].isna().sum()
    )

    return report