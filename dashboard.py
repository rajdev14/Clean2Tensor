import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "best_no2_lstm.pth"
PREPROCESSING_FILE = BASE_DIR / "preprocessing.pkl"
HISTORY_FILE = BASE_DIR / "training_history.csv"
PREDICTIONS_FILE = BASE_DIR / "test_predictions.csv"
DATA_FILE = BASE_DIR / "data" / "raw" / "AirQualityUCI.csv"

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
    def __init__(self, input_size=9, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None

    model = NO2LSTM()

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu"
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


@st.cache_data
def load_preprocessing():
    if not PREPROCESSING_FILE.exists():
        return None

    with open(PREPROCESSING_FILE, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_history():
    if not HISTORY_FILE.exists():
        return None

    return pd.read_csv(HISTORY_FILE)


@st.cache_data
def load_predictions():
    if not PREDICTIONS_FILE.exists():
        return None

    return pd.read_csv(PREDICTIONS_FILE)


def read_raw_data():

    if not DATA_FILE.exists():
        return None

    try:
        df = pd.read_csv(
            DATA_FILE,
            sep=";",
            decimal=","
        )

        if len(df.columns) <= 2:
            df = pd.read_csv(DATA_FILE)

    except Exception:
        df = pd.read_csv(DATA_FILE)

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    if "Date" in df.columns and "Time" in df.columns:

        df["Timestamp"] = pd.to_datetime(
            df["Date"].astype(str).str.strip()
            + " "
            + df["Time"].astype(str).str.strip(),
            format="%d/%m/%Y %H.%M.%S",
            errors="coerce"
        )

    for column in df.columns:

        if column not in [
            "Date",
            "Time",
            "Timestamp"
        ]:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df = df.replace(-200, np.nan)

    return df


def get_preprocessing_values(preprocessing):

    medians = preprocessing.get("medians")
    means = preprocessing.get("feature_means")
    stds = preprocessing.get("feature_stds")

    target_mean = preprocessing.get("target_mean")
    target_std = preprocessing.get("target_std")

    if medians is None:
        raise ValueError(
            "Training medians are missing from preprocessing.pkl"
        )

    if means is None:
        raise ValueError(
            "Feature means are missing from preprocessing.pkl"
        )

    if stds is None:
        raise ValueError(
            "Feature standard deviations are missing from preprocessing.pkl"
        )

    if target_mean is None or target_std is None:
        raise ValueError(
            "Target scaling values are missing from preprocessing.pkl"
        )

    median_values = pd.Series(
        medians,
        dtype=float
    ).reindex(FEATURES)

    mean_values = pd.Series(
        means,
        dtype=float
    ).reindex(FEATURES)

    std_values = pd.Series(
        stds,
        dtype=float
    ).reindex(FEATURES)

    if median_values.isna().any():
        raise ValueError(
            "Some feature medians are missing."
        )

    if mean_values.isna().any():
        raise ValueError(
            "Some feature means are missing."
        )

    if std_values.isna().any():
        raise ValueError(
            "Some feature standard deviations are missing."
        )

    return (
        median_values,
        mean_values,
        std_values,
        float(target_mean),
        float(target_std)
    )


def predict_latest():

    preprocessing = load_preprocessing()
    model = load_model()
    data = read_raw_data()

    if preprocessing is None:
        raise FileNotFoundError(
            "preprocessing.pkl was not found."
        )

    if model is None:
        raise FileNotFoundError(
            "best_no2_lstm.pth was not found."
        )

    if data is None:
        raise FileNotFoundError(
            "AirQualityUCI.csv was not found."
        )

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing features: "
            + ", ".join(missing_features)
        )

    if "Timestamp" not in data.columns:
        raise ValueError(
            "Timestamp column was not created."
        )

    data = data.dropna(
        subset=["Timestamp"]
    )

    data = data.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    (
        medians,
        means,
        stds,
        target_mean,
        target_std
    ) = get_preprocessing_values(
        preprocessing
    )

    latest = data[
        FEATURES
    ].tail(
        SEQUENCE_LENGTH
    ).copy()

    if len(latest) < SEQUENCE_LENGTH:
        raise ValueError(
            "Less than 24 observations are available."
        )

    latest = latest.fillna(
        medians
    )

    scaled = (
        latest - means
    ) / stds

    tensor = torch.tensor(
        scaled.to_numpy(
            dtype=np.float32
        ),
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        scaled_prediction = model(
            tensor
        ).item()

    prediction = (
        scaled_prediction * target_std
        + target_mean
    )

    timestamp = data[
        "Timestamp"
    ].iloc[-1]

    return (
        float(prediction),
        timestamp,
        latest
    )


def find_prediction_columns(df):

    actual_column = None
    predicted_column = None

    for column in df.columns:

        name = str(column).lower()
        name = name.replace("_", " ")
        name = name.strip()

        if (
            "actual" in name
            and "no2" in name
        ):
            actual_column = column

        if (
            "predicted" in name
            and "no2" in name
        ):
            predicted_column = column

    if actual_column is None:

        for column in df.columns:

            name = str(column).lower()

            if name in [
                "actual",
                "actual no2"
            ]:
                actual_column = column

    if predicted_column is None:

        for column in df.columns:

            name = str(column).lower()

            if name in [
                "predicted",
                "predicted no2"
            ]:
                predicted_column = column

    return (
        actual_column,
        predicted_column
    )


st.set_page_config(
    page_title="Clean2Tensor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.7;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">🌫️ Clean2Tensor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Air Quality Data Cleaning, LSTM Training and NO₂ Prediction'
    '</div>',
    unsafe_allow_html=True
)


with st.sidebar:

    st.header("System Status")

    files = {
        "AirQualityUCI.csv": DATA_FILE,
        "LSTM Model": MODEL_FILE,
        "Preprocessing": PREPROCESSING_FILE,
        "Training History": HISTORY_FILE,
        "Test Predictions": PREDICTIONS_FILE
    }

    for name, path in files.items():

        if path.exists():

            st.success(
                name + " available"
            )

        else:

            st.error(
                name + " missing"
            )

    st.divider()

    st.write(
        "**Framework:** Streamlit"
    )

    st.write(
        "**Deep Learning:** PyTorch"
    )

    st.write(
        "**Model:** 2-Layer LSTM"
    )

    st.write(
        "**Sequence:** 24 hours"
    )


history = load_history()
predictions = load_predictions()


actual_column = None
predicted_column = None

metrics = None


if predictions is not None:

    (
        actual_column,
        predicted_column
    ) = find_prediction_columns(
        predictions
    )

    if (
        actual_column is not None
        and predicted_column is not None
    ):

        actual = pd.to_numeric(
            predictions[actual_column],
            errors="coerce"
        )

        predicted = pd.to_numeric(
            predictions[predicted_column],
            errors="coerce"
        )

        valid = (
            actual.notna()
            & predicted.notna()
        )

        actual = actual[valid]
        predicted = predicted[valid]

        if len(actual) > 0:

            mae = np.mean(
                np.abs(
                    actual - predicted
                )
            )

            rmse = np.sqrt(
                np.mean(
                    (actual - predicted) ** 2
                )
            )

            ss_res = np.sum(
                (actual - predicted) ** 2
            )

            ss_tot = np.sum(
                (actual - actual.mean()) ** 2
            )

            r2 = (
                1 - ss_res / ss_tot
                if ss_tot != 0
                else np.nan
            )

            metrics = {
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "Samples": len(actual)
            }


st.subheader("Clean2Tensor Overview")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Model",
        "2-Layer LSTM"
    )


with col2:

    st.metric(
        "Input Features",
        "9"
    )


with col3:

    st.metric(
        "Sequence Length",
        "24 Hours"
    )


with col4:

    if metrics:

        st.metric(
            "Test Samples",
            f"{metrics['Samples']:,}"
        )

    else:

        st.metric(
            "Test Samples",
            "N/A"
        )


if metrics:

    st.subheader(
        "Test Performance"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "MAE",
            f"{metrics['MAE']:.4f} µg/m³"
        )

    with col2:

        st.metric(
            "RMSE",
            f"{metrics['RMSE']:.4f} µg/m³"
        )

    with col3:

        st.metric(
            "R²",
            f"{metrics['R2']:.4f}"
        )


st.divider()


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Performance",
        "🔮 Prediction",
        "📈 Dataset",
        "🤖 Model"
    ]
)


with tab1:

    st.header(
        "Model Performance"
    )

    if history is not None:

        st.subheader(
            "Training and Validation Loss"
        )

        history_copy = history.copy()

        train_column = None
        validation_column = None

        for column in history_copy.columns:

            name = str(column).lower()

            if (
                "train" in name
                and "loss" in name
            ):
                train_column = column

            if (
                "val" in name
                and "loss" in name
            ):
                validation_column = column

        if (
            train_column is not None
            and validation_column is not None
        ):

            loss_data = history_copy[
                [
                    train_column,
                    validation_column
                ]
            ].copy()

            loss_data.columns = [
                "Training Loss",
                "Validation Loss"
            ]

            st.line_chart(
                loss_data,
                height=400
            )

        else:

            st.dataframe(
                history,
                use_container_width=True
            )

    else:

        st.warning(
            "training_history.csv was not found."
        )


    if (
        predictions is not None
        and actual_column is not None
        and predicted_column is not None
    ):

        st.subheader(
            "Actual vs Predicted NO₂"
        )

        chart_data = predictions.copy()

        timestamp_column = None

        for column in chart_data.columns:

            name = str(column).lower()

            if (
                "timestamp" in name
                or "time" in name
            ):

                timestamp_column = column
                break

        chart_data[
            actual_column
        ] = pd.to_numeric(
            chart_data[actual_column],
            errors="coerce"
        )

        chart_data[
            predicted_column
        ] = pd.to_numeric(
            chart_data[predicted_column],
            errors="coerce"
        )

        chart_data = chart_data.dropna(
            subset=[
                actual_column,
                predicted_column
            ]
        )

        if timestamp_column is not None:

            chart_data[
                timestamp_column
            ] = pd.to_datetime(
                chart_data[timestamp_column],
                errors="coerce"
            )

            chart_data = chart_data.dropna(
                subset=[
                    timestamp_column
                ]
            )

            chart_data = chart_data.set_index(
                timestamp_column
            )

        comparison = chart_data[
            [
                actual_column,
                predicted_column
            ]
        ].copy()

        comparison.columns = [
            "Actual NO₂",
            "Predicted NO₂"
        ]

        st.line_chart(
            comparison,
            height=450
        )


        st.subheader(
            "Prediction Error"
        )

        error = (
            comparison["Actual NO₂"]
            - comparison["Predicted NO₂"]
        )

        error_data = pd.DataFrame(
            {
                "Prediction Error":
                error
            }
        )

        st.line_chart(
            error_data,
            height=300
        )

    else:

        st.warning(
            "test_predictions.csv does not contain "
            "recognizable actual/predicted columns."
        )


with tab2:

    st.header(
        "NO₂ Prediction"
    )

    st.write(
        "The prediction uses the latest 24 observations, "
        "the saved training preprocessing values and "
        "the trained LSTM model."
    )

    if st.button(
        "🔮 Predict Latest NO₂",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Preparing data and running the LSTM..."
            ):

                prediction, timestamp, latest_data = (
                    predict_latest()
                )

            st.success(
                "Prediction completed successfully."
            )

            st.metric(
                "Predicted NO₂ Concentration",
                f"{prediction:.2f} µg/m³"
            )

            st.write(
                f"Latest observation: **{timestamp}**"
            )

            st.subheader(
                "Prediction Input"
            )

            display_data = latest_data.copy()

            display_data.index = range(
                1,
                len(display_data) + 1
            )

            st.dataframe(
                display_data,
                use_container_width=True
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


with tab3:

    st.header(
        "Air Quality Dataset"
    )

    raw_data = read_raw_data()

    if raw_data is not None:

        valid_timestamps = (
            raw_data["Timestamp"]
            .notna()
            .sum()
            if "Timestamp" in raw_data.columns
            else 0
        )

        missing_target = (
            raw_data["NO2(GT)"]
            .isna()
            .sum()
            if "NO2(GT)" in raw_data.columns
            else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Rows",
                f"{len(raw_data):,}"
            )

        with col2:

            st.metric(
                "Columns",
                f"{len(raw_data.columns):,}"
            )

        with col3:

            st.metric(
                "Valid Timestamps",
                f"{valid_timestamps:,}"
            )

        with col4:

            st.metric(
                "Missing NO₂",
                f"{missing_target:,}"
            )


        st.subheader(
            "Selected Input Features"
        )

        feature_table = pd.DataFrame(
            {
                "Feature": FEATURES,
                "Type": ["Input"] * len(FEATURES)
            }
        )

        st.dataframe(
            feature_table,
            use_container_width=True,
            hide_index=True
        )


        st.subheader(
            "Input Feature Missingness"
        )

        missing_table = pd.DataFrame(
            {
                "Feature": FEATURES,
                "Missing Count": [
                    int(
                        raw_data[feature]
                        .isna()
                        .sum()
                    )
                    for feature in FEATURES
                ]
            }
        )

        missing_table[
            "Missing Percentage"
        ] = (
            missing_table["Missing Count"]
            / len(raw_data)
            * 100
        ).round(2)

        st.dataframe(
            missing_table,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.error(
            "AirQualityUCI.csv was not found."
        )


with tab4:

    st.header(
        "LSTM Model"
    )

    model = load_model()

    if model is not None:

        st.subheader(
            "Architecture"
        )

        st.code(
            str(model),
            language="text"
        )

        total_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total Parameters",
                f"{total_parameters:,}"
            )

        with col2:

            st.metric(
                "Trainable Parameters",
                f"{trainable_parameters:,}"
            )

        with col3:

            st.metric(
                "Device",
                "CPU"
            )


    st.subheader(
        "Input Features"
    )

    for feature in FEATURES:

        st.write(
            f"• {feature}"
        )


    st.subheader(
        "Clean2Tensor Pipeline"
    )

    st.code(
        """
Raw Air Quality Dataset
        ↓
Timestamp Validation
        ↓
Invalid Value Handling
        ↓
Feature Selection
        ↓
Chronological Train / Validation / Test Split
        ↓
Training-Median Imputation
        ↓
Training-Based Feature Scaling
        ↓
24-Hour Sequence Creation
        ↓
PyTorch Dataset
        ↓
2-Layer LSTM
        ↓
NO₂ Prediction
        """,
        language="text"
    )


    st.info(
        "Imputation and feature scaling parameters are calculated "
        "from the training data and reused during prediction."
    )


st.divider()

st.caption(
    "Clean2Tensor • Air Quality Data Cleaning and NO₂ Forecasting • "
    "PyTorch + Streamlit"
)