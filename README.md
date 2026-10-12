# 🎓 AI Student Performance Predictor

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/sardaressa456-boop/ai-student-performance-predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/sardaressa456-boop/ai-student-performance-predictor/actions)

An end-to-end Machine Learning web application and analytics platform that analyzes student academic, behavioral, and demographic attributes to predict final secondary examination performance (`G3`, 0–20 scale).

Designed with an **early-intervention philosophy**: the model deliberately excludes mid-term period grades (`G1`, `G2`) to give educators and advisors actionable foresight before report cards are finalized.

![AI Student Performance Predictor Dashboard](assets/dashboard.png)

- **GitHub Repository**: [https://github.com/sardaressa456-boop/ai-student-performance-predictor](https://github.com/sardaressa456-boop/ai-student-performance-predictor)
- **Author Profile**: [sardaressa456-boop](https://github.com/sardaressa456-boop)
- **Live Streamlit App**: [Deploy via Streamlit Community Cloud](https://share.streamlit.io/)

---

## 📌 Project Overview

Academic underperformance often stems from compound factors—inadequate study time, past failures, lack of educational support, and home environment. This project builds a reliable, reproducible machine learning system from the ground up:

1. **Exploratory Data Analysis (EDA)**: Investigates grade distributions, correlations, and feature impacts on 395 secondary students.
2. **Leakage-Free Preprocessing**: Leverages Scikit-learn `Pipeline` and `ColumnTransformer` to handle imputation, categorical one-hot encoding, and numerical standardization strictly within training folds.
3. **Rigorous Model Selection**: Compares Baseline Mean, Linear Regression, and Random Forest using 5-fold cross-validation.
4. **Interactive Dashboard**: A modern, dark-navy Streamlit web application with Plotly analytics, interactive grade prediction, dynamic performance tiers, and CSV export.
5. **Continuous Integration**: Automated test suite with `pytest` and GitHub Actions testing pipeline on every push.

---

## 🏗️ Repository Architecture

```text
ai-student-performance-predictor/
├── .github/
│   └── workflows/
│       └── tests.yml             # Automated CI pipeline with GitHub Actions
├── .streamlit/
│   └── config.toml               # Professional dark navy theme configuration
├── assets/
│   ├── dashboard.png             # UI overview dashboard screenshot
│   ├── actual_vs_predicted.png   # Test set evaluation scatter plot
│   ├── grade_distribution.png    # Target grade distribution chart
│   └── studytime_vs_grade.png    # Average grade vs. study time chart
├── data/
│   └── student-mat.csv           # UCI Student Performance Dataset (Math)
├── models/
│   ├── model_comparison.csv      # 5-fold cross-validation metrics
│   ├── model_metadata.json       # Serialized training metadata & test scores
│   └── student_model.joblib      # Production Scikit-learn pipeline artifact
├── notebooks/
│   └── exploratory_analysis.ipynb# Interactive Jupyter notebook for EDA
├── src/
│   ├── __init__.py
│   ├── explore_data.py           # Exploratory analysis script & static chart generator
│   ├── preprocess.py             # Feature selection, cleaning, & Pipeline definition
│   ├── train.py                  # 5-fold CV comparison, winner selection & persistence
│   └── evaluate.py               # Independent test set evaluation & diagnostic plot
├── tests/
│   ├── test_pipeline.py          # Pipeline and model validation tests
│   ├── test_app.py               # Streamlit AppTest interface smoke test
│   └── test_prediction.py        # Dataset & inference test suite
├── app.py                        # Multi-page Streamlit web application
├── requirements.txt              # Exact pinned project dependencies
├── LICENSE                       # Open-source MIT License
└── README.md                     # Project documentation
```

---

## 📊 Dataset & Features

Sourced from the **[UCI Machine Learning Repository: Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)** (Cortez and Silva, 2008), licensed under CC BY 4.0.

### Input Features (10 Variables)
| Feature | Type | Description |
| :--- | :--- | :--- |
| `age` | Numeric | Student age (15–22) |
| `studytime` | Ordinal | Weekly study time (1: <2h, 2: 2–5h, 3: 5–10h, 4: >10h) |
| `failures` | Numeric | Past class failures (0–4) |
| `traveltime` | Ordinal | Home-to-school travel time (1: <15m, 2: 15–30m, 3: 30–60m, 4: >60m) |
| `schoolsup` | Categorical | Extra educational support from school (`yes` / `no`) |
| `famsup` | Categorical | Family educational support (`yes` / `no`) |
| `paid` | Categorical | Extra paid classes in mathematics (`yes` / `no`) |
| `activities` | Categorical | Extracurricular activities (`yes` / `no`) |
| `higher` | Categorical | Desires to pursue higher education (`yes` / `no`) |
| `internet` | Categorical | Internet access at home (`yes` / `no`) |

*Target*: **`G3`** (Final mathematics exam grade, 0 to 20).

---

## 🔬 Model Benchmarking & Results

Models were evaluated on **316 training records** using **5-fold cross-validation** and finally verified on **79 held-out test records**.

### 5-Fold Cross-Validation Comparison

| Model | CV MAE (pts) | CV RMSE (pts) | CV R² Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression (Selected)** | **3.2159** | **4.2688** | **0.0878** |
| Random Forest Regressor | 3.3864 | 4.4920 | -0.0208 |
| Baseline (Mean Regressor) | 3.4029 | 4.5635 | -0.0350 |

* **Selection Decision**: Linear Regression demonstrated the lowest average prediction error across all five validation folds while avoiding the overfitting tendencies observed with tree ensembles on this sample size.
* **Held-out Test Performance (79 unseen students)**:
  * **MAE**: `3.7495` grade points
  * **RMSE**: `4.5905` grade points
  * **R² Score**: `-0.0277`

![Actual vs Predicted Grades](assets/actual_vs_predicted.png)

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/sardaressa456-boop/ai-student-performance-predictor.git
cd ai-student-performance-predictor

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install exact pinned dependencies
pip install -r requirements.txt
```

### 2. Run Preprocessing & Training
```bash
# Run exploratory data analysis and generate static charts
python src/explore_data.py

# Run cross-validation and save the best model artifact
python -m src.train

# Evaluate model on the 79 held-out test students
python -m src.evaluate
```

### 3. Run Automated Tests
```bash
python -m pytest -v
```

### 4. Launch the Interactive Web Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🌟 Application Features

1. **Executive Dashboard**: KPI summary cards (records, features, test MAE), interactive distribution histograms with mean/median reference lines.
2. **AI Prediction Engine**: Dynamic profile builder that uses the fitted Scikit-learn Pipeline directly, categorizing predictions into performance tiers (Distinction, Good Standing, Pass, At-Risk) with CSV result exports.
3. **Interactive Data Explorer**: Filter students by study habits and aspirations, examine failure impacts with box plots, and inspect support interactions.
4. **Model Performance Hub**: In-depth residual analysis, cross-validation bar charts, and transparent error metric reporting.

---

## ⚖️ Limitations & Ethical Considerations

* **Sample Size**: The dataset comprises 395 students across two Portuguese secondary schools in 2005–2006. Regional and cultural context may differ from modern educational environments.
* **Early Prediction Trade-off**: Omitting mid-term grades (`G1`, `G2`) lowers correlation and increases error variance, but provides earlier intervention opportunities.
* **Educational Ethics**: This application is intended solely as an exploratory academic tool. Automated scores must never replace human academic advising, holistic student evaluations, or individualized support plans.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
The dataset is provided under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://archive.ics.uci.edu/dataset/320/student+performance).
