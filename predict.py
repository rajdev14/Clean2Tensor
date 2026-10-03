from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "best_no2_lstm.pth"
SCALER_FILE = BASE_DIR / "preprocessing.pkl"

FEATURES = [
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

SEQUENCE_LENGTH = 24


class NO2LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=9,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        return self.fc(output)


def load_preprocessing():
    if not SCALER_FILE.exists():
        raise FileNotFoundError(
            f"{SCALER_FILE} not found."
        )

    with open(SCALER_FILE, "rb") as file:
        preprocessing = pickle.load(file)

    return preprocessing


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"{MODEL_FILE} not found."
        )

    model = NO2LSTM()

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model


def get_preprocessing_values(preprocessing):

    required_keys = [
        "features",
        "target",
        "sequence_length",
        "medians",
        "feature_means",
        "feature_stds",
        "target_mean",
        "target_std"
    ]

    for key in required_keys:
        if key not in preprocessing:
            raise ValueError(
                f"{key} not found in preprocessing.pkl"
            )

    features = preprocessing["features"]
    target = preprocessing["target"]
    sequence_length = preprocessing["sequence_length"]
    medians = preprocessing["medians"]
    feature_means = preprocessing["feature_means"]
    feature_stds = preprocessing["feature_stds"]
    target_mean = preprocessing["target_mean"]
    target_std = preprocessing["target_std"]

    return (
        features,
        target,
        sequence_length,
        medians,
        feature_means,
        feature_stds,
        target_mean,
        target_std
    )


def prepare_input(data, preprocessing):

    (
        features,
        target,
        sequence_length,
        medians,
        feature_means,
        feature_stds,
        target_mean,
        target_std
    ) = get_preprocessing_values(preprocessing)

    df = pd.DataFrame(data)

    missing_columns = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing features: "
            + ", ".join(missing_columns)
        )

    df = df[features].copy()

    for feature in features:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

        df[feature] = df[feature].fillna(
            medians[feature]
        )

    for feature in features:

        df[feature] = (
            df[feature] - feature_means[feature]
        ) / feature_stds[feature]

    values = df[features].values.astype(
        np.float32
    )

    if len(values) != sequence_length:
        raise ValueError(
            f"Exactly {sequence_length} hourly "
            f"observations are required. "
            f"Received {len(values)}."
        )

    tensor = torch.tensor(
        values,
        dtype=torch.float32
    ).unsqueeze(0)

    return (
        tensor,
        target_mean,
        target_std
    )


def predict(
    model,
    tensor,
    target_mean,
    target_std
):

    with torch.no_grad():

        prediction = model(tensor)

    prediction = prediction.item()

    prediction = (
        prediction * target_std
        + target_mean
    )

    return prediction


def load_sample_from_csv():

    csv_file = (
        BASE_DIR
        / "data"
        / "raw"
        / "AirQualityUCI.csv"
    )

    if not csv_file.exists():
        raise FileNotFoundError(
            f"{csv_file} not found."
        )

    df = pd.read_csv(
        csv_file,
        sep=";",
        decimal=","
    )

    df.columns = [
        column.strip()
        for column in df.columns
    ]

    df = df.replace(
        -200,
        np.nan
    )

    missing_columns = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing: "
            + ", ".join(missing_columns)
        )

    df = df[FEATURES].copy()

    df = df.tail(
        SEQUENCE_LENGTH
    )

    return df


def main():

    print("======================================")
    print("       CLEAN2TENSOR PREDICTION")
    print("======================================")

    print("\nLoading preprocessing data...")

    preprocessing = load_preprocessing()

    print(
        "Preprocessing data loaded successfully."
    )

    print("\nLoading trained model...")

    model = load_model()

    print(
        "Model loaded successfully."
    )

    print(
        "\nLoading latest 24 observations..."
    )

    data = load_sample_from_csv()

    print(
        f"Input sequence shape: "
        f"{data.shape}"
    )

    print("\nPreparing input...")

    (
        tensor,
        target_mean,
        target_std
    ) = prepare_input(
        data,
        preprocessing
    )

    print(
        f"Tensor shape: "
        f"{tensor.shape}"
    )

    print("\nRunning prediction...")

    predicted_no2 = predict(
        model,
        tensor,
        target_mean,
        target_std
    )

    print("\n======================================")
    print("          PREDICTION RESULT")
    print("======================================")

    print(
        f"Predicted NO2 concentration: "
        f"{predicted_no2:.2f} µg/m³"
    )

    print("======================================")

    print(
        "\nPrediction completed successfully."
    )


if __name__ == "__main__":
    main()