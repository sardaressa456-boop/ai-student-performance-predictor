import sys
from pathlib import Path

# Ensure project root is in sys.path so running directly via VS Code or terminal works
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

from sklearn.model_selection import train_test_split
from src.preprocess import load_student_data, FEATURES, TARGET

MODEL_PATH = BASE_DIR / "models" / "student_model.joblib"
ASSETS_DIR = BASE_DIR / "assets"


def main():
    ASSETS_DIR.mkdir(exist_ok=True)

    # 1. Load data and recreate original test split
    df = load_student_data()
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run python -m src.train first!")

    # 2. Load trained pipeline
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n==========================================")
    print("      HELD-OUT TEST SET EVALUATION        ")
    print("==========================================")
    print(f"Test Records: {len(X_test)}")
    print(f"MAE:          {mae:.4f} grade points")
    print(f"RMSE:         {rmse:.4f} grade points")
    print(f"R² Score:     {r2:.4f}")
    print("==========================================\n")

    print("Sample 5 Predictions vs. Real Grades:")
    samples_df = pd.DataFrame({
        "Actual G3": y_test.values[:5],
        "Predicted G3": y_pred[:5].round(2),
        "Absolute Error": (abs(y_test.values[:5] - y_pred[:5])).round(2)
    })
    print(samples_df.to_string(index=False))

    # 3. Create actual vs. predicted visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(
        y_test,
        y_pred,
        alpha=0.75,
        color="#2563eb",
        edgecolors="#1e3a8a",
        s=65,
        label="Test Students"
    )
    plt.plot([0, 20], [0, 20], color="#dc2626", linestyle="--", lw=2, label="Perfect Prediction (y = x)")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.title("Actual vs. Predicted Final Exam Grades (G3)", fontsize=14, fontweight="bold")
    plt.xlabel("Actual Grade (0 - 20)", fontsize=12)
    plt.ylabel("Predicted Grade (0 - 20)", fontsize=12)
    plt.legend(frameon=True, loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    chart_path = ASSETS_DIR / "actual_vs_predicted.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\nEvaluation chart saved successfully to: {chart_path}")

    # 4. Confirm raw feature inference
    sample_student = X_test.iloc[[0]]
    sample_prediction = model.predict(sample_student)[0]
    print(f"Verification: Single raw student input prediction = {sample_prediction:.2f} / 20")


if __name__ == "__main__":
    main()
