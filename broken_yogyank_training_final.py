
"""
Yogyank Entitlement Score

Author: Awantika Srivastava
Version: v1.0
"""

import os
import json
import joblib
import warnings
import pandas as pd
import numpy as np

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

warnings.filterwarnings("ignore")


# =========================================================
# CONFIG
# =========================================================

DATA_PATH = "farmer_scoring_sample_yogyank_round1_final.csv"

TARGET_COLUMN = "target_entitlement_score"

MODEL_VERSION = "v2.0"

ARTIFACT_DIR = "artifacts"


# =========================================================
# EXPECTED SCHEMA
# =========================================================

EXPECTED_COLUMNS = [
    "land_area_acres",
    "crop_type",
    "pm_kisan_status",
    "historical_repayment_score",
    TARGET_COLUMN
]


# =========================================================
# LOAD DATA
# =========================================================

def load_dataset(path):

    print("\nLoading dataset...")

    df = pd.read_csv(path)

    print(f"Dataset shape: {df.shape}")

    return df


# =========================================================
# VALIDATE SCHEMA
# =========================================================

def validate_schema(df):

    print("\nValidating dataset schema...")

    missing_cols = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}"
        )

    print("Schema validation passed.")


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def prepare_features(df):

    print("\nPreparing safe features...")

    # -----------------------------------------------------
    # REMOVE LEAKAGE FEATURES
    # -----------------------------------------------------

    leakage_features = [
        "defaulted_in_next_12_months"
    ]

    existing_leakage = [
        col for col in leakage_features
        if col in df.columns
    ]

    if existing_leakage:
        print(f"Removed leakage features: {existing_leakage}")

    # -----------------------------------------------------
    # SAFE FEATURES
    # -----------------------------------------------------

    safe_features = []

    for col in df.columns:

        if col in leakage_features:
            continue

        if col == TARGET_COLUMN:
            continue

        safe_features.append(col)

    X = df[safe_features].copy()

    y = df[TARGET_COLUMN].copy()

    print(f"Total safe features used: {len(safe_features)}")

    return X, y


# =========================================================
# BUILD PREPROCESSOR
# =========================================================

def build_preprocessor(X):

    print("\nBuilding preprocessing pipeline...")

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print(f"Numeric features count: {len(numeric_features)}")
    print(f"Categorical features count: {len(categorical_features)}")

    # -----------------------------------------------------
    # NUMERIC PIPELINE
    # -----------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # -----------------------------------------------------
    # CATEGORICAL PIPELINE
    # -----------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # -----------------------------------------------------
    # COLUMN TRANSFORMER
    # -----------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# =========================================================
# BUILD MODEL
# =========================================================

def build_model_pipeline(preprocessor):

    print("\nBuilding model pipeline...")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    return pipeline


# =========================================================
# SPLIT DATA
# =========================================================

def split_data(X, y):

    """
    Deterministic validation split.
    No shuffling.
    """

    print("\nCreating validation split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        shuffle=False,
        random_state=42
    )

    print(f"Train rows: {len(X_train)}")
    print(f"Validation rows: {len(X_test)}")

    return X_train, X_test, y_train, y_test


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(
    pipeline,
    X_train,
    y_train
):

    print("\nTraining model...")

    pipeline.fit(
        X_train,
        y_train
    )

    print("Model training completed.")

    return pipeline


# =========================================================
# EVALUATE MODEL
# =========================================================

def evaluate_model(
    pipeline,
    X_test,
    y_test
):

    print("\nRunning validation...")

    predictions = pipeline.predict(X_test)

    r2 = r2_score(
        y_test,
        predictions
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    metrics = {

        "r2_score": round(r2, 4),

        "mae": round(mae, 4),

        "rmse": round(rmse, 4)
    }

    print("\nValidation Metrics")
    print(metrics)

    return metrics


# =========================================================
# FAIRNESS MONITORING
# =========================================================

def fairness_monitoring(df):

    print("\n==============================")
    print("FAIRNESS MONITORING")
    print("==============================")

    if "pm_kisan_status" in df.columns:

        print("\nAverage Target by PM-Kisan Status")

        print(
            df.groupby(
                "pm_kisan_status"
            )[TARGET_COLUMN].mean()
        )

    if "crop_type" in df.columns:

        print("\nAverage Target by Crop Type")

        print(
            df.groupby(
                "crop_type"
            )[TARGET_COLUMN].mean().head(10)
        )


# =========================================================
# REASON CODES
# =========================================================

def generate_reason_codes(
    pipeline,
    top_n=3
):

    print("\nGenerating reason codes...")

    model = pipeline.named_steps["model"]

    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    importances = model.feature_importances_

    importance_df = pd.DataFrame({

        "feature": feature_names,

        "importance": importances
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    top_features = importance_df.head(top_n)

    reason_codes = top_features[
        "feature"
    ].tolist()

    print("\nTop Reason Codes:")

    for idx, reason in enumerate(reason_codes, start=1):

        print(f"{idx}. {reason}")

    return reason_codes


# =========================================================
# SAVE ARTIFACTS
# =========================================================

def save_artifacts(
    pipeline,
    feature_names,
    metrics,
    reason_codes
):

    print("\nSaving artifacts...")

    os.makedirs(
        ARTIFACT_DIR,
        exist_ok=True
    )

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model_path = os.path.join(
        ARTIFACT_DIR,
        "yogyank_pipeline.pkl"
    )

    joblib.dump(
        pipeline,
        model_path
    )

    # -----------------------------------------------------
    # FEATURE SCHEMA
    # -----------------------------------------------------

    schema_path = os.path.join(
        ARTIFACT_DIR,
        "feature_schema.json"
    )

    with open(schema_path, "w") as f:

        json.dump(
            feature_names,
            f,
            indent=4
        )

    # -----------------------------------------------------
    # METADATA
    # -----------------------------------------------------

    metadata = {

        "model_version": MODEL_VERSION,

        "created_at": datetime.utcnow().isoformat(),

        "target_column": TARGET_COLUMN,

        "feature_count": len(feature_names),

        "features_used": feature_names,

        "metrics": metrics,

        "top_reason_codes": reason_codes
    }

    metadata_path = os.path.join(
        ARTIFACT_DIR,
        "metadata.json"
    )

    with open(metadata_path, "w") as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    # -----------------------------------------------------
    # ARTIFACT MANIFEST
    # -----------------------------------------------------

    manifest = {

        "model_file": "yogyank_pipeline.pkl",

        "schema_file": "feature_schema.json",

        "metadata_file": "metadata.json",

        "version": MODEL_VERSION
    }

    manifest_path = os.path.join(
        ARTIFACT_DIR,
        "artifact_manifest.json"
    )

    with open(manifest_path, "w") as f:

        json.dump(
            manifest,
            f,
            indent=4
        )

    print(f"Artifacts saved in: {ARTIFACT_DIR}/")


# =========================================================
# SCORING FUNCTION
# =========================================================

def score_farmer(input_df):

    """
    Example inference function.
    """

    model_path = os.path.join(
        ARTIFACT_DIR,
        "yogyank_pipeline.pkl"
    )

    pipeline = joblib.load(model_path)

    predictions = pipeline.predict(input_df)

    return predictions


# =========================================================
# MAIN FLOW
# =========================================================

def main():

    print("\n====================================")
    print("YOGYANK SAFE TRAINING PIPELINE v2.0")
    print("====================================")

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    df = load_dataset(DATA_PATH)

    # -----------------------------------------------------
    # VALIDATE
    # -----------------------------------------------------

    validate_schema(df)

    # -----------------------------------------------------
    # FAIRNESS CHECKS
    # -----------------------------------------------------

    fairness_monitoring(df)

    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------

    X, y = prepare_features(df)

    # -----------------------------------------------------
    # PREPROCESSOR
    # -----------------------------------------------------

    preprocessor = build_preprocessor(X)

    # -----------------------------------------------------
    # PIPELINE
    # -----------------------------------------------------

    pipeline = build_model_pipeline(
        preprocessor
    )

    # -----------------------------------------------------
    # SPLIT
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    pipeline = train_model(
        pipeline,
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # EVALUATE
    # -----------------------------------------------------

    metrics = evaluate_model(
        pipeline,
        X_test,
        y_test
    )

    # -----------------------------------------------------
    # REASON CODES
    # -----------------------------------------------------

    reason_codes = generate_reason_codes(
        pipeline
    )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    save_artifacts(
        pipeline,
        X.columns.tolist(),
        metrics,
        reason_codes
    )

    print("\n====================================")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("====================================")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()