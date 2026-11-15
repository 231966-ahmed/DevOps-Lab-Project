
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load Random Forest, label encoder and feature names
with open(BASE_DIR / "other_model_data.pkl", "rb") as file:
    model_data = pickle.load(file)

random_forest = model_data["random_forest"]
label_encoder = model_data["label_encoder"]
feature_names = model_data["feature_names"]

# Load XGBoost separately
xgboost_model = xgb.XGBClassifier()
xgboost_model.load_model(BASE_DIR / "xgboost_model.json")

# Temporary fallback: used only if training medians
# have not yet been saved with the model data.
training_medians = model_data.get("medians")


def prepare_data(df):
    """Prepare uploaded CSV data for model prediction."""

    # Normalize column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Remove label column if the uploaded CSV contains it
    if "label" in df.columns:
        df = df.drop(columns=["label"])

    # Check for missing required features
    missing = [col for col in feature_names if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep features in the exact training order
    df = df[feature_names].copy()

    # Convert features to numeric values
    for col in feature_names:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Handle missing values
    if df.isnull().any().any():
        if training_medians is not None:
            df = df.fillna(training_medians)
        else:
            # Temporary fallback until original training medians are added
            df = df.fillna(df.median()).fillna(0)

    return df


def predict(df, model_name):
    """Generate predictions using the selected model."""

    prepared_data = prepare_data(df)

    if model_name == "random_forest":
        model = random_forest
    elif model_name == "xgboost":
        model = xgboost_model
    else:
        raise ValueError("Invalid model. Choose random_forest or xgboost.")

    predictions = model.predict(prepared_data)

    # Convert encoded predictions into original attack names
    predicted_labels = label_encoder.inverse_transform(
        predictions.astype(int)
    )

    return predicted_labels.tolist()
