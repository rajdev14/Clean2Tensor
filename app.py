import os
import json
import random
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "raw" / "AirQualityUCI.csv"
MODEL_FILE = BASE_DIR / "best_no2_lstm.pth"
SCALER_FILE = "preprocessing.pkl"
HISTORY_FILE = "training_history.csv"
PREDICTION_FILE = "test_predictions.csv"
LOSS_PLOT = "training_loss.png"

TARGET = "NO2(GT)"
FEATURES = [
    "PT08.S1(CO)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]

SEQ_LEN = 24
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 8

def load_data():
    print("Loading dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE, sep=";", decimal=",", encoding="latin1")

    print("Dataset loaded successfully.")
    return df

def audit_raw(df):
    print("\n===== RAW DATA AUDIT =====")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    timestamp = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce",
    )

    print(f"Valid timestamps: {timestamp.notna().sum()}")
    print(f"Invalid timestamps: {timestamp.isna().sum()}")
    print(f"Duplicate timestamps: {timestamp.duplicated().sum()}")
    print(f"Completely empty rows: {df.isna().all(axis=1).sum()}")

def clean_data(df):
    print("\n===== CLEANING DATA =====")

    df = df.copy()

    timestamp = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce",
    )

    df["Timestamp"] = timestamp

    numeric_columns = [c for c in df.columns if c not in ["Date", "Time", "Timestamp"]]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_columns] = df[numeric_columns].replace(-200, np.nan)

    before = len(df)
    df = df.dropna(subset=["Timestamp"])
    after_timestamp = len(df)

    duplicate_count = df["Timestamp"].duplicated().sum()
    df = df.drop_duplicates(subset=["Timestamp"])

    df = df.sort_values("Timestamp").reset_index(drop=True)

    print(f"Initial rows: {before}")
    print(f"Rows after timestamp validation: {after_timestamp}")
    print(f"Invalid timestamp rows removed: {before - after_timestamp}")
    print(f"Duplicate timestamps removed: {duplicate_count}")
    print(f"Final rows: {len(df)}")
    print(f"Missing {TARGET} targets: {df[TARGET].isna().sum()}")

    return df

def feature_selection(df):
    print("\n===== FEATURE SELECTION =====")

    required = FEATURES + [TARGET, "Timestamp"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    model_df = df[required].copy()

    for feature in FEATURES + [TARGET]:
        model_df[feature] = pd.to_numeric(model_df[feature], errors="coerce")

    print("Selected features:")
    for feature in FEATURES:
        print(f"- {feature}")
    print(f"- {TARGET}")
    print("- Timestamp")
    print(f"\nModel dataset shape: {model_df.shape}")

    return model_df

def chronological_split(df):
    print("\n===== CHRONOLOGICAL SPLIT =====")

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, val, test

def missing_analysis(df):
    print("\n===== MISSING INPUT ANALYSIS =====")

    rows = []
    for feature in FEATURES:
        count = int(df[feature].isna().sum())
        rows.append(
            {
                "feature": feature,
                "missing_count": count,
                "missing_percentage": round(count / len(df) * 100, 2),
            }
        )

    report = pd.DataFrame(rows)

    print("\nFeature missingness:")
    print(report.to_string(index=False))

    input_missing = df[FEATURES].isna().any(axis=1)
    all_available = df[FEATURES].notna().all(axis=1)
    multiple_missing = df[FEATURES].isna().sum(axis=1) > 1

    print(f"\nRows with any missing input: {input_missing.sum()}")
    print(f"Rows with all inputs available: {all_available.sum()}")
    print(f"Rows with multiple missing inputs: {multiple_missing.sum()}")
    print(f"Missing {TARGET} targets: {df[TARGET].isna().sum()}")

def impute_features(train, val, test):
    print("\n===== MISSING VALUE PREPROCESSING =====")

    train = train.copy()
    val = val.copy()
    test = test.copy()

    medians = train[FEATURES].median()

    train[FEATURES] = train[FEATURES].fillna(medians)
    val[FEATURES] = val[FEATURES].fillna(medians)
    test[FEATURES] = test[FEATURES].fillna(medians)

    print("\nTraining medians used for imputation:")
    print(medians.to_string())

    print("\nRemaining missing input values:")
    print(f"Training: {train[FEATURES].isna().sum().sum()}")
    print(f"Validation: {val[FEATURES].isna().sum().sum()}")
    print(f"Testing: {test[FEATURES].isna().sum().sum()}")

    return train, val, test, medians

def scale_features(train, val, test):
    print("\n===== FEATURE SCALING =====")

    means = train[FEATURES].mean()
    stds = train[FEATURES].std(ddof=0).replace(0, 1.0)

    train = train.copy()
    val = val.copy()
    test = test.copy()

    train[FEATURES] = (train[FEATURES] - means) / stds
    val[FEATURES] = (val[FEATURES] - means) / stds
    test[FEATURES] = (test[FEATURES] - means) / stds

    print("\nTraining means:")
    print(means.to_string())

    print("\nTraining standard deviations:")
    print(stds.to_string())

    print("\nScaled training feature means:")
    print(train[FEATURES].mean().round(6).to_string())

    print("\nScaled training feature standard deviations:")
    print(train[FEATURES].std(ddof=0).round(6).to_string())

    return train, val, test, means, stds

def create_sequences(df):
    X = []
    y = []
    timestamps = []

    values = df[FEATURES].to_numpy(dtype=np.float32)
    targets = df[TARGET].to_numpy(dtype=np.float32)

    for i in range(SEQ_LEN, len(df)):
        if np.isnan(targets[i]):
            continue

        X.append(values[i - SEQ_LEN:i])
        y.append(targets[i])
        timestamps.append(df["Timestamp"].iloc[i])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        pd.to_datetime(timestamps),
    )

def create_dataloaders(train, val, test):
    print("\n===== SEQUENCE CREATION =====")

    X_train, y_train, t_train = create_sequences(train)
    X_val, y_val, t_val = create_sequences(val)
    X_test, y_test, t_test = create_sequences(test)

    print("\n===== PYTORCH DATASET CREATION =====")
    print(f"Training dataset size: {len(X_train)}")
    print(f"Validation dataset size: {len(X_val)}")
    print(f"Testing dataset size: {len(X_test)}")

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\n===== PYTORCH DATALOADERS =====")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Testing batches: {len(test_loader)}")

    xb, yb = next(iter(train_loader))

    print("\n===== BATCH VERIFICATION =====")
    print(f"X batch shape: {xb.shape}")
    print(f"y batch shape: {yb.shape}")
    print(f"X batch dtype: {xb.dtype}")
    print(f"y batch dtype: {yb.dtype}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    return (
        train_loader,
        val_loader,
        test_loader,
        X_train,
        y_train,
        t_train,
        X_val,
        y_val,
        t_val,
        X_test,
        y_test,
        t_test,
    )

class NO2LSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])

def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            loss = criterion(pred, y)

            total_loss += loss.item() * len(X)
            count += len(X)

    return total_loss / count

def train_model(model, train_loader, val_loader, device, target_mean, target_std):
    print("\n===== MODEL TRAINING =====")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    target_mean_t = torch.tensor(target_mean, dtype=torch.float32, device=device)
    target_std_t = torch.tensor(target_std, dtype=torch.float32, device=device)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            prediction = model(X)
            loss = criterion(prediction, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(X)
            train_count += len(X)

        train_loss = train_loss_sum / train_count
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        train_rmse_original = np.sqrt(train_loss) * target_std
        val_rmse_original = np.sqrt(val_loss) * target_std

        history.append(
            {
                "epoch": epoch,
                "training_loss_scaled": train_loss,
                "validation_loss_scaled": val_loss,
                "training_rmse": train_rmse_original,
                "validation_rmse": val_rmse_original,
                "learning_rate": current_lr,
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Training Loss: {train_loss:.6f} | "
            f"Validation Loss: {val_loss:.6f} | "
            f"Validation RMSE: {val_rmse_original:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_FILE)
            print("  Best model saved.")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(HISTORY_FILE, index=False)

    print(f"\nBest validation loss: {best_val_loss}")
    print(f"Best model: {MODEL_FILE}")

    return history_df, best_val_loss

def evaluate_model(model, test_loader, device, target_mean, target_std, timestamps):
    print("\n===== LOADING BEST MODEL =====")

    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    predictions_scaled = []
    actual_scaled = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)

            pred = model(X).cpu().numpy().reshape(-1)
            actual = y.numpy().reshape(-1)

            predictions_scaled.extend(pred)
            actual_scaled.extend(actual)

    predictions_scaled = np.asarray(predictions_scaled)
    actual_scaled = np.asarray(actual_scaled)

    predictions = predictions_scaled * target_std + target_mean
    actual = actual_scaled * target_std + target_mean

    mae = np.mean(np.abs(actual - predictions))
    rmse = np.sqrt(np.mean((actual - predictions) ** 2))

    ss_res = np.sum((actual - predictions) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    results = pd.DataFrame(
        {
            "Timestamp": timestamps[:len(actual)],
            "Actual_NO2": actual,
            "Predicted_NO2": predictions,
        }
    )

    results.to_csv(PREDICTION_FILE, index=False)

    print("\n===== TEST RESULTS =====")
    print(f"Test samples: {len(actual)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    print("\n===== SAMPLE PREDICTIONS =====")
    print(f"{'Timestamp':<25}{'Actual NO2':>15}{'Predicted NO2':>20}")

    for i in range(min(10, len(results))):
        print(
            f"{str(results.iloc[i]['Timestamp']):<25}"
            f"{results.iloc[i]['Actual_NO2']:>15.2f}"
            f"{results.iloc[i]['Predicted_NO2']:>20.2f}"
        )

    return results, mae, rmse, r2

def save_preprocessing(medians, means, stds, target_mean, target_std):
    data = {
        "features": FEATURES,
        "target": TARGET,
        "sequence_length": SEQ_LEN,
        "medians": medians.to_dict(),
        "feature_means": means.to_dict(),
        "feature_stds": stds.to_dict(),
        "target_mean": float(target_mean),
        "target_std": float(target_std),
    }

    with open(SCALER_FILE, "wb") as f:
        pickle.dump(data, f)

def plot_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(
        history["epoch"],
        history["training_loss_scaled"],
        label="Training Loss",
    )
    plt.plot(
        history["epoch"],
        history["validation_loss_scaled"],
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (scaled target)")
    plt.title("NO2 LSTM Training History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT, dpi=150)
    plt.close()

def print_final_summary(
    model,
    train,
    val,
    test,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    mae,
    rmse,
    r2,
):
    print("\n===== TRAINING SUMMARY =====")
    print(f"Training rows: {len(train)}")
    print(f"Validation rows: {len(val)}")
    print(f"Testing rows: {len(test)}")

    print(f"\nTraining sequences: X {X_train.shape}, y {y_train.shape}")
    print(f"Validation sequences: X {X_val.shape}, y {y_val.shape}")
    print(f"Testing sequences: X {X_test.shape}, y {y_test.shape}")

    print("\n===== SPLIT TIME RANGES =====")
    print(
        f"Training: {train['Timestamp'].min()} to {train['Timestamp'].max()}"
    )
    print(
        f"Validation: {val['Timestamp'].min()} to {val['Timestamp'].max()}"
    )
    print(
        f"Testing: {test['Timestamp'].min()} to {test['Timestamp'].max()}"
    )

    print("\n===== FINAL TEST METRICS =====")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    print("\n===== OUTPUT FILES =====")
    print(f"- {MODEL_FILE}")
    print(f"- {SCALER_FILE}")
    print(f"- {HISTORY_FILE}")
    print(f"- {PREDICTION_FILE}")
    print(f"- {LOSS_PLOT}")

def main():
    print("======================================")
    print("         CLEAN2TENSOR")
    print(" DATA INGESTION AND CLEANING")
    print("======================================")

    df_raw = load_data()
    audit_raw(df_raw)

    df_clean = clean_data(df_raw)
    model_df = feature_selection(df_clean)

    missing_analysis(model_df)

    train, val, test = chronological_split(model_df)

    train, val, test, medians = impute_features(train, val, test)

    train, val, test, feature_means, feature_stds = scale_features(
        train, val, test
    )

    valid_target_train = train[TARGET].dropna()

    if len(valid_target_train) == 0:
        raise ValueError("No valid NO2 target values are available in training data.")

    target_mean = valid_target_train.mean()
    target_std = valid_target_train.std(ddof=0)

    if target_std == 0:
        target_std = 1.0

    train[TARGET] = (train[TARGET] - target_mean) / target_std
    val[TARGET] = (val[TARGET] - target_mean) / target_std
    test[TARGET] = (test[TARGET] - target_mean) / target_std

    print("\n===== TARGET SCALING =====")
    print(f"Training NO2 mean: {target_mean:.6f}")
    print(f"Training NO2 standard deviation: {target_std:.6f}")

    (
        train_loader,
        val_loader,
        test_loader,
        X_train,
        y_train,
        t_train,
        X_val,
        y_val,
        t_val,
        X_test,
        y_test,
        t_test,
    ) = create_dataloaders(train, val, test)

    print("\n===== MODEL CREATION =====")

    model = NO2LSTM(
        input_size=len(FEATURES),
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(model)
    print(f"\nDevice: {device}")

    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"\nTotal parameters: {total_parameters}")
    print(f"Trainable parameters: {trainable_parameters}")

    print("\n===== MODEL FORWARD PASS =====")

    sample_x, _ = next(iter(train_loader))
    sample_x = sample_x.to(device)

    with torch.no_grad():
        sample_output = model(sample_x)

    print(f"Input shape: {sample_x.shape}")
    print(f"Output shape: {sample_output.shape}")

    save_preprocessing(
        medians,
        feature_means,
        feature_stds,
        target_mean,
        target_std,
    )

    history, best_val_loss = train_model(
        model,
        train_loader,
        val_loader,
        device,
        target_mean,
        target_std,
    )

    plot_history(history)

    results, mae, rmse, r2 = evaluate_model(
        model,
        test_loader,
        device,
        target_mean,
        target_std,
        t_test,
    )

    print_final_summary(
        model,
        train,
        val,
        test,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        mae,
        rmse,
        r2,
    )

    print("\n===== CLEAN2TENSOR COMPLETED SUCCESSFULLY =====")

if __name__ == "__main__":
    main()
