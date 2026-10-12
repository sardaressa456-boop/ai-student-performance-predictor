import sys
from pathlib import Path

# Project root directory and sys.path registration
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Dataset location
DATA_PATH = BASE_DIR / "data" / "student-mat.csv"

# Numerical features
NUMERIC_FEATURES = [
    "age",
    "studytime",
    "failures",
    "traveltime"
]

# Categorical features
CATEGORICAL_FEATURES = [
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "higher",
    "internet"
]

# All input features and prediction target
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "G3"


def load_student_data():
    """Load, validate, and clean student records from CSV."""
    df = pd.read_csv(DATA_PATH, sep=";")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Check required columns
    required = FEATURES + [TARGET]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")

    # Convert numeric features to numbers
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean categorical values
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].apply(
            lambda value: value.strip().lower()
            if isinstance(value, str)
            else value
        )
        # Treat empty strings as missing
        df[col] = df[col].replace("", float("nan"))

    # Convert target to numeric
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    return df


def build_preprocessor():
    """Build a reusable preprocessing ColumnTransformer pipeline."""
    # Numerical preprocessing pipeline
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing pipeline
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    # Combine both pipelines into a ColumnTransformer
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return preprocessor


if __name__ == "__main__":
    # Load and clean data
    df = load_student_data()
    print("\nOriginal dataset shape:", df.shape)

    # Select features and target
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Split before fitting preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # Build preprocessing system
    preprocessor = build_preprocessor()

    # Learn preprocessing only from training data (no data leakage)
    X_train_ready = preprocessor.fit_transform(X_train)

    # Apply the learned transformations to test data
    X_test_ready = preprocessor.transform(X_test)

    # Names of final encoded columns
    feature_names = preprocessor.get_feature_names_out()

    print("\nProcessed training shape:", X_train_ready.shape)
    print("Processed testing shape:", X_test_ready.shape)
    print("Total engineered feature columns:", len(feature_names))
    print("\nPreprocessing completed successfully!")
