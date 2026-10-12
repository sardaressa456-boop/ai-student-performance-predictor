import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is in sys.path so running directly via VS Code or terminal works
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

from src.preprocess import (
    load_student_data,
    build_preprocessor,
    FEATURES,
    TARGET
)

MODEL_DIR = BASE_DIR / "models"


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    # 1. Load the cleaned student data
    df = load_student_data()
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # 2. Split train/test (80/20) with random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training students: {len(X_train)}")
    print(f"Held-out test students: {len(X_test)}")

    # 3. Define candidate models
    models = {
        "Baseline (Mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42
        )
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2"
    }

    results = []
    fitted_pipelines = {}

    print("\nFive-fold training cross-validation:")
    for name, regressor in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor)
        ])
        cv_res = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
        mae = -cv_res["test_mae"].mean()
        rmse = -cv_res["test_rmse"].mean()
        r2 = cv_res["test_r2"].mean()

        print(f"  {name:18s} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
        results.append({
            "Model": name,
            "CV_MAE": round(mae, 4),
            "CV_RMSE": round(rmse, 4),
            "CV_R2": round(r2, 4)
        })

        # Fit full pipeline on all training records
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

    # 4. Save model comparison
    comparison_df = pd.DataFrame(results)
    comparison_csv_path = MODEL_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\nModel comparison saved to: {comparison_csv_path}")

    # 5. Model selection rule: Lowest CV MAE
    best_row = comparison_df.loc[comparison_df["CV_MAE"].idxmin()]
    selected_name = best_row["Model"]
    print(f"Selected model: {selected_name}")

    best_pipeline = fitted_pipelines[selected_name]
    model_save_path = MODEL_DIR / "student_model.joblib"
    joblib.dump(best_pipeline, model_save_path)
    print(f"Saved trained pipeline to: {model_save_path}")

    # 6. Evaluate selected model on untouched test set
    y_pred_test = best_pipeline.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = root_mean_squared_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    # 7. Save metadata
    metadata = {
        "selected_model": selected_name,
        "features": FEATURES,
        "target": TARGET,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_size": 0.20,
        "random_state": 42,
        "cv_folds": 5,
        "cv_mae": float(best_row["CV_MAE"]),
        "cv_rmse": float(best_row["CV_RMSE"]),
        "cv_r2": float(best_row["CV_R2"]),
        "test_mae": round(float(test_mae), 4),
        "test_rmse": round(float(test_rmse), 4),
        "test_r2": round(float(test_r2), 4),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat()
    }

    metadata_path = MODEL_DIR / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model metadata saved to: {metadata_path}")
    print("\nTest data was not used for model selection.")


if __name__ == "__main__":
    main()
