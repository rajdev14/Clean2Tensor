import pandas as pd


def load_air_quality_data(file_path):
    df = pd.read_csv(
        file_path,
        sep=";",
        decimal=","
    )

    df = df.drop(
        columns=["Unnamed: 15", "Unnamed: 16"],
        errors="ignore"
    )

    df["Timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " +
        df["Time"].astype(str).str.replace(".", ":", regex=False),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )

    return df