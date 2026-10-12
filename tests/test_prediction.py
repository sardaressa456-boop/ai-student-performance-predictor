"""
Unit tests for AI Student Performance Predictor.
"""
import sys
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.preprocess import load_student_data, build_preprocessor, FEATURES, TARGET

MODEL_PATH = BASE_DIR / "models" / "student_model.joblib"


def test_dataset_loading():
    """Verify student dataset loads with expected dimensions and columns."""
    df = load_student_data()
    assert df.shape == (395, 33), f"Expected shape (395, 33), got {df.shape}"
    for col in FEATURES + [TARGET]:
        assert col in df.columns, f"Expected column {col} missing in dataset"


def test_preprocessor_transformation():
    """Verify preprocessor produces 16 features from the 10 raw input features."""
    df = load_student_data()
    X = df[FEATURES]
    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(X)
    assert X_trans.shape == (395, 16), f"Expected (395, 16), got {X_trans.shape}"


def test_model_file_exists():
    """Ensure trained model artifact exists on disk."""
    assert MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}"


def test_model_inference():
    """Ensure trained pipeline accepts raw feature inputs without manual encoding."""
    model = joblib.load(MODEL_PATH)
    sample_student = pd.DataFrame([{
        "age": 16,
        "studytime": 3,
        "failures": 0,
        "traveltime": 1,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "higher": "yes",
        "internet": "yes"
    }])
    pred = model.predict(sample_student)
    assert len(pred) == 1
    assert isinstance(float(pred[0]), float)
    # The grade scale is 0 to 20, predictions should reasonably be within or near this range
    assert -5.0 <= pred[0] <= 25.0, f"Unreasonable prediction value: {pred[0]}"


if __name__ == "__main__":
    test_dataset_loading()
    test_preprocessor_transformation()
    test_model_file_exists()
    test_model_inference()
    print("All unit tests passed successfully!")
