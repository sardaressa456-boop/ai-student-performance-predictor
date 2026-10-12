import sys
from pathlib import Path
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import load_student_data, build_preprocessor, FEATURES, TARGET

MODEL_FILE = ROOT / "models" / "student_model.joblib"


def test_dataset_validity():
    df = load_student_data()
    assert TARGET in df.columns
    assert df[TARGET].notna().all()
    assert df[TARGET].between(0, 20).all()


def test_preprocessing():
    df = load_student_data()
    X = df[FEATURES].iloc[:100]
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == 100
    assert transformed.shape[1] >= len(FEATURES)
    assert np.isfinite(transformed).all()


def test_unseen_category():
    df = load_student_data()
    X = df[FEATURES].iloc[:100]
    preprocessor = build_preprocessor()
    preprocessor.fit(X)
    sample = X.iloc[[0]].copy()
    sample.loc[sample.index[0], "schoolsup"] = "unknown"
    result = preprocessor.transform(sample)
    assert np.isfinite(result).all()


def test_saved_model_prediction():
    assert MODEL_FILE.exists(), f"Model file not found at {MODEL_FILE}"
    model = joblib.load(MODEL_FILE)
    df = load_student_data()
    sample = df[FEATURES].iloc[[0]]
    prediction = model.predict(sample)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()
