"""
🎓 AI Student Performance Predictor - Streamlit Dashboard
Interactive Machine Learning Application for Educational Grade Prediction
"""

import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (Dark Navy Glassmorphism Theme)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.4) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #93c5fd, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    /* Stat Cards */
    .stat-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(71, 85, 105, 0.4);
        border-radius: 12px;
        padding: 18px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.5);
    }
    
    .stat-label {
        font-size: 0.825rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    
    .stat-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .stat-subtext {
        font-size: 0.775rem;
        color: #38bdf8;
        margin-top: 4px;
    }
    
    /* Result Card */
    .prediction-card {
        background: linear-gradient(145deg, rgba(30, 58, 138, 0.35), rgba(15, 23, 42, 0.85));
        border: 1px solid rgba(96, 165, 250, 0.4);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
        margin: 20px 0;
    }
    
    .prediction-score {
        font-size: 3.5rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 10px 0;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
    }
    
    .tier-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Paths & Cached Loaders
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student-mat.csv"
MODEL_PATH = BASE_DIR / "models" / "student_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
COMPARISON_PATH = BASE_DIR / "models" / "model_comparison.csv"

FEATURES = [
    "age", "studytime", "failures", "traveltime",
    "schoolsup", "famsup", "paid", "activities", "higher", "internet"
]


@st.cache_data
def load_dataset():
    """Load the UCI student mathematics dataset."""
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH, sep=";")


@st.cache_resource
def load_trained_model():
    """Load the trained Scikit-learn Pipeline."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    """Load model evaluation metadata and cross-validation comparisons."""
    meta = None
    comp = None
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    if COMPARISON_PATH.exists():
        comp = pd.read_csv(COMPARISON_PATH)
    return meta, comp


df_raw = load_dataset()
model = load_trained_model()
metadata, comparison_df = load_metadata()

# ---------------------------------------------------------
# Sidebar Navigation & Details
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 Navigation")
    selected_page = st.radio(
        "Go to page:",
        ["📊 Overview", "🔮 Predict Grade", "📈 Data Explorer", "🔬 Model Performance"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("#### ℹ️ Project Context")
    st.caption("**Dataset**: UCI Student Performance (Mathematics)")
    st.caption("**Target**: Final Exam Grade (`G3`, 0–20 scale)")
    st.caption("**Early Prediction**: Period grades (G1, G2) are intentionally omitted for early-warning analysis.")

    if metadata:
        st.markdown("---")
        st.markdown("#### 🏆 Active Model")
        st.success(f"**{metadata.get('selected_model', 'Linear Regression')}**")
        st.caption(f"Trained on: {metadata.get('train_samples', 316)} students")
        st.caption(f"Held-out test: {metadata.get('test_samples', 79)} students")

    st.markdown("---")
    st.caption("Developed with Python, Scikit-learn, and Streamlit.")

# ---------------------------------------------------------
# PAGE 1: OVERVIEW
# ---------------------------------------------------------
if selected_page == "📊 Overview":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🎓 AI Student Performance Predictor</div>
        <div class="hero-subtitle">
            An end-to-end Machine Learning web system built to predict final secondary examination performance 
            from study habits, attendance factors, past academic challenges, and family support mechanisms.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Stat Metric Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Dataset Records</div>
            <div class="stat-value">395</div>
            <div class="stat-subtext">Real Portuguese math students</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Input Features</div>
            <div class="stat-value">10</div>
            <div class="stat-subtext">4 numeric + 6 categorical</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        test_mae = metadata.get("test_mae", 3.75) if metadata else 3.75
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Test Set MAE</div>
            <div class="stat-value">{test_mae:.2f}</div>
            <div class="stat-subtext">Mean absolute error (0–20 pts)</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        winner = metadata.get("selected_model", "Linear Regression") if metadata else "Linear Regression"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Selected Model</div>
            <div class="stat-value" style="font-size: 1.35rem; margin-top: 4px;">{winner}</div>
            <div class="stat-subtext">5-Fold CV lowest MAE winner</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Visual: Distribution of Final Grades
    if df_raw is not None:
        st.subheader("📊 Distribution of Student Final Exam Grades (G3)")
        
        fig_dist = px.histogram(
            df_raw,
            x="G3",
            nbins=21,
            color_discrete_sequence=["#3b82f6"],
            labels={"G3": "Final Exam Grade (0 - 20)", "count": "Student Count"},
            opacity=0.85
        )
        
        avg_grade = df_raw["G3"].mean()
        med_grade = df_raw["G3"].median()

        fig_dist.add_vline(
            x=avg_grade,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Mean: {avg_grade:.2f}",
            annotation_position="top left"
        )
        fig_dist.add_vline(
            x=med_grade,
            line_dash="dot",
            line_color="#10b981",
            annotation_text=f"Median: {med_grade:.1f}",
            annotation_position="top right"
        )
        
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", range=[-0.5, 20.5]),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)

    # Architecture & Key Insights
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🛠️ Architecture & Pipeline")
        st.markdown("""
        * **Early Warning Focus**: Excludes mid-term grades `G1` and `G2` so schools can identify students needing support *before* report cards.
        * **Standardized Transformations**: Built with Scikit-learn `Pipeline` and `ColumnTransformer` (median imputation, scaling, and one-hot encoding).
        * **Strict Leakage Prevention**: Preprocessors and encoders are fitted strictly within cross-validation folds and training sets.
        """)
    with col_b:
        st.markdown("### ⚠️ Educational Disclaimer")
        st.info("""
        **Learning Demonstration Only**:
        This predictive model is trained on a historical educational dataset of 395 students. In practice, academic success is shaped by numerous nuanced emotional, socio-economic, and personal factors not captured here. Predictions should be interpreted as approximate indicators, not deterministic assessments.
        """)

# ---------------------------------------------------------
# PAGE 2: PREDICT GRADE
# ---------------------------------------------------------
elif selected_page == "🔮 Predict Grade":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🔮 Predict Student Final Grade</div>
        <div class="hero-subtitle">
            Configure a student profile across demographic, study-habit, and support variables 
            to generate an AI estimation of their final mathematics grade.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Trained model not found! Please run `python -m src.train` first.")
    else:
        with st.form("prediction_form"):
            st.markdown("#### 📝 Student Profile Inputs")

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("##### 📚 Academic & Study Habits")
                age = st.slider("Student Age", min_value=15, max_value=22, value=16, step=1)
                
                studytime_options = {
                    1: "< 2 hours / week",
                    2: "2 to 5 hours / week",
                    3: "5 to 10 hours / week",
                    4: "> 10 hours / week"
                }
                studytime = st.selectbox(
                    "Weekly Study Time",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: f"{x}: {studytime_options[x]}",
                    index=1
                )

                failures = st.slider(
                    "Previous Class Failures",
                    min_value=0, max_value=4, value=0,
                    help="Number of past class failures (0 to 4)"
                )

                traveltime_options = {
                    1: "< 15 minutes",
                    2: "15 to 30 minutes",
                    3: "30 to 60 minutes",
                    4: "> 60 minutes"
                }
                traveltime = st.selectbox(
                    "Home-to-School Travel Time",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: f"{x}: {traveltime_options[x]}",
                    index=0
                )

            with c2:
                st.markdown("##### 🤝 Support & Environment")
                schoolsup = st.selectbox("Extra Educational Support from School (`schoolsup`)", ["no", "yes"], index=0)
                famsup = st.selectbox("Family Educational Support (`famsup`)", ["yes", "no"], index=0)
                paid = st.selectbox("Extra Paid Mathematics Classes (`paid`)", ["no", "yes"], index=0)
                activities = st.selectbox("Extracurricular Activities (`activities`)", ["yes", "no"], index=0)
                higher = st.selectbox("Wants to Pursue Higher Education (`higher`)", ["yes", "no"], index=0)
                internet = st.selectbox("Internet Access at Home (`internet`)", ["yes", "no"], index=0)

            submitted = st.form_submit_button("🚀 Generate Prediction", use_container_width=True)

        if submitted:
            # Build input dataframe
            student_dict = {
                "age": age,
                "studytime": studytime,
                "failures": failures,
                "traveltime": traveltime,
                "schoolsup": schoolsup,
                "famsup": famsup,
                "paid": paid,
                "activities": activities,
                "higher": higher,
                "internet": internet
            }

            student_frame = pd.DataFrame([student_dict], columns=FEATURES)
            raw_prediction = float(model.predict(student_frame)[0])
            bounded_prediction = max(0.0, min(20.0, raw_prediction))

            # Performance Tier
            if bounded_prediction >= 16:
                tier = "🌟 Distinction (16–20)"
                badge_bg = "background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981;"
                tip = "Excellent performance trajectory! Encourage advanced materials and peer mentorship."
            elif bounded_prediction >= 14:
                tier = "✨ Good Standing (14–15)"
                badge_bg = "background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #3b82f6;"
                tip = "Solid performance. Targeted reinforcement on challenging topics can elevate this to distinction."
            elif bounded_prediction >= 10:
                tier = "📘 Satisfactory / Pass (10–13)"
                badge_bg = "background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b;"
                tip = "Passing grade. Additional weekly study time and tutorial sessions recommended."
            else:
                tier = "⚠️ At-Risk of Failing (<10)"
                badge_bg = "background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444;"
                tip = "High risk of academic struggle. Early proactive intervention, counseling, and tutoring strongly advised."

            # Result Card
            st.markdown(f"""
            <div class="prediction-card">
                <span class="tier-badge" style="{badge_bg}">{tier}</span>
                <div style="color: #94a3b8; font-size: 1rem; margin-top: 4px;">Predicted Final Mathematics Grade (G3)</div>
                <div class="prediction-score">{bounded_prediction:.2f} <span style="font-size: 1.5rem; color: #94a3b8;">/ 20</span></div>
                <div style="color: #cbd5e1; font-size: 0.95rem; max-width: 600px; margin: 0 auto;">
                    {tip}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if raw_prediction < 0.0 or raw_prediction > 20.0:
                st.caption(f"ℹ️ Note: Linear regression raw calculated value was {raw_prediction:.2f}, bounded to the official 0–20 Portuguese exam scale.")

            # Gauge Meter
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bounded_prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 20], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#38bdf8"},
                    'bgcolor': "#1e293b",
                    'borderwidth': 1,
                    'bordercolor': "#475569",
                    'steps': [
                        {'range': [0, 10], 'color': 'rgba(239, 68, 68, 0.25)'},
                        {'range': [10, 14], 'color': 'rgba(245, 158, 11, 0.25)'},
                        {'range': [14, 16], 'color': 'rgba(59, 130, 246, 0.25)'},
                        {'range': [16, 20], 'color': 'rgba(16, 185, 129, 0.25)'}
                    ],
                    'threshold': {
                        'line': {'color': "#ffffff", 'width': 3},
                        'thickness': 0.75,
                        'value': bounded_prediction
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "#f8fafc", 'family': "Inter"},
                height=240,
                margin=dict(l=20, r=20, t=10, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # CSV Export
            export_df = student_frame.copy()
            export_df["predicted_G3"] = round(bounded_prediction, 2)
            export_df["raw_predicted_G3"] = round(raw_prediction, 2)
            export_df["performance_tier"] = tier

            csv_data = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Prediction Result as CSV",
                data=csv_data,
                file_name=f"student_prediction_{bounded_prediction:.1f}.csv",
                mime="text/csv",
                use_container_width=True
            )

# ---------------------------------------------------------
# PAGE 3: DATA EXPLORER
# ---------------------------------------------------------
elif selected_page == "📈 Data Explorer":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">📈 Data Analytics & Exploration</div>
        <div class="hero-subtitle">
            Interactively explore relationships between study time, previous failures, and academic achievement.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df_raw is None:
        st.error("Dataset not loaded!")
    else:
        # Filters in columns
        st.markdown("#### 🔍 Filter Exploration Data")
        fcol1, fcol2, fcol3 = st.columns(3)

        with fcol1:
            sel_study = st.multiselect(
                "Filter Study Time Categories:",
                options=[1, 2, 3, 4],
                default=[1, 2, 3, 4],
                format_func=lambda x: f"Level {x}"
            )
        with fcol2:
            sel_higher = st.multiselect(
                "Wants Higher Education:",
                options=["yes", "no"],
                default=["yes", "no"]
            )
        with fcol3:
            max_fail = st.slider("Max Previous Failures Allowed:", 0, 4, 4)

        filtered_df = df_raw[
            (df_raw["studytime"].isin(sel_study)) &
            (df_raw["higher"].isin(sel_higher)) &
            (df_raw["failures"] <= max_fail)
        ]

        st.caption(f"Showing **{len(filtered_df)}** of {len(df_raw)} student records based on filters.")

        # Visualizations in 2 Columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### ⏱️ Average Grade by Study Time")
            study_summary = (
                filtered_df.groupby("studytime", as_index=False)["G3"]
                .agg(mean_g3="mean", count="count")
            )
            study_summary["studytime_label"] = study_summary["studytime"].map({
                1: "1: <2h", 2: "2: 2-5h", 3: "3: 5-10h", 4: "4: >10h"
            })

            fig_study = px.bar(
                study_summary,
                x="studytime_label",
                y="mean_g3",
                color="mean_g3",
                color_continuous_scale="Blues",
                labels={"studytime_label": "Study Time Category", "mean_g3": "Average Grade"},
                text=study_summary["mean_g3"].round(2)
            )
            fig_study.update_layout(
                yaxis_range=[0, 20],
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                font=dict(color="#e2e8f0"),
                showlegend=False
            )
            st.plotly_chart(fig_study, use_container_width=True)

        with col2:
            st.markdown("##### 📉 Impact of Past Class Failures")
            fig_fail = px.box(
                filtered_df,
                x="failures",
                y="G3",
                color="failures",
                labels={"failures": "Past Failures Count", "G3": "Final Grade (0–20)"},
                color_discrete_sequence=px.colors.sequential.Sunsetdark
            )
            fig_fail.update_layout(
                yaxis_range=[-0.5, 20.5],
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                font=dict(color="#e2e8f0"),
                showlegend=False
            )
            st.plotly_chart(fig_fail, use_container_width=True)

        # Support vs Grade Heatmap / Grouped Chart
        st.markdown("##### 🤝 Family Support & School Support Cross-Comparison")
        support_group = filtered_df.groupby(["schoolsup", "famsup"], as_index=False)["G3"].mean()
        fig_sup = px.bar(
            support_group,
            x="schoolsup",
            y="G3",
            color="famsup",
            barmode="group",
            labels={"schoolsup": "School Support", "famsup": "Family Support", "G3": "Average Grade"},
            color_discrete_map={"yes": "#3b82f6", "no": "#f97316"}
        )
        fig_sup.update_layout(
            yaxis_range=[0, 20],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(color="#e2e8f0")
        )
        st.plotly_chart(fig_sup, use_container_width=True)

        # Raw Table View
        with st.expander("📄 View Filtered Data Table"):
            st.dataframe(filtered_df[FEATURES + ["G3"]], use_container_width=True)

# ---------------------------------------------------------
# PAGE 4: MODEL PERFORMANCE
# ---------------------------------------------------------
elif selected_page == "🔬 Model Performance":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🔬 Model Comparison & Evaluation</div>
        <div class="hero-subtitle">
            Cross-validation benchmark results, held-out test set performance, and residual analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if comparison_df is not None:
        st.subheader("📊 5-Fold Cross-Validation Comparison")
        st.markdown("Candidate models evaluated on 316 training samples using 5-fold cross-validation:")

        st.dataframe(comparison_df.style.highlight_min(subset=["CV_MAE", "CV_RMSE"], color="#1e3a8a"), use_container_width=True)

        # Bar chart comparison
        fig_comp = px.bar(
            comparison_df,
            x="Model",
            y="CV_MAE",
            color="Model",
            labels={"CV_MAE": "Cross-Validation MAE (Lower is Better)"},
            text=comparison_df["CV_MAE"].round(4),
            color_discrete_sequence=["#94a3b8", "#3b82f6", "#10b981"]
        )
        fig_comp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(color="#e2e8f0"),
            showlegend=False
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Held-out Test Metrics
    st.markdown("---")
    st.subheader("🎯 Held-Out Test Set Metrics (79 Unseen Students)")

    col1, col2, col3 = st.columns(3)
    if metadata:
        with col1:
            st.metric("Test MAE", f"{metadata.get('test_mae', 0):.2f} pts", help="Mean Absolute Error on 79 test records")
        with col2:
            st.metric("Test RMSE", f"{metadata.get('test_rmse', 0):.2f} pts", help="Root Mean Squared Error on 79 test records")
        with col3:
            st.metric("Test R² Score", f"{metadata.get('test_r2', 0):.4f}", help="Coefficient of determination")

    # Actual vs Predicted Plot
    eval_chart_path = BASE_DIR / "assets" / "actual_vs_predicted.png"
    if eval_chart_path.exists():
        st.markdown("##### 📉 Actual vs. Predicted Grades Scatter Plot")
        st.image(str(eval_chart_path), caption="Evaluated on 79 held-out test students", use_container_width=True)

    st.markdown("---")
    st.markdown("### 💡 Key Findings & Discussion")
    st.markdown("""
    1. **Linear Regression Selected**: Through 5-fold cross-validation, Linear Regression achieved the lowest average MAE (`3.22`), outperforming Random Forest (`3.39`) and the Mean Baseline (`3.40`).
    2. **Avoiding Data Leakage**: Preprocessing (imputation, scaling, one-hot encoding) was learned strictly within training folds, ensuring test set isolation.
    3. **Generalization Constraints**: Because the dataset contains only 395 records from two specific schools, test R² is near 0 without period grades (G1 and G2). This reinforces the honest machine learning principle: predicting complex academic outcomes early requires richer behavioral context.
    """)
