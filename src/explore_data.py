import sys
from pathlib import Path
# Locate the project's root folder and ensure it is in sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = BASE_DIR / "data" / "student-mat.csv"
df = pd.read_csv(file_path, sep=";")

# Display the first five rows
print("\nFIRST FIVE STUDENTS")
print(df.head())

# Dataset dimensions
print("\nDATASET SHAPE")
print(df.shape)

# Column names
print("\nCOLUMN NAMES")
print(df.columns.tolist())

# Dataset data types and completeness
print("\nDATASET INFORMATION")
df.info()

# Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Duplicate rows
print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

# Final grade statistics
print("\nFINAL GRADE ANALYSIS")
print("Average Grade:", round(df["G3"].mean(), 2))
print("Highest Grade:", df["G3"].max())
print("Lowest Grade:", df["G3"].min())

# Study time vs final grade
print("\nAVERAGE GRADE BY STUDY TIME")
print(df.groupby("studytime")["G3"].mean().round(2))

# Relationship between numeric columns and G3
print("\nCORRELATION WITH FINAL GRADE")
numeric_df = df.select_dtypes(include="number")
print(numeric_df.corr()["G3"].sort_values(ascending=False))

# Create an output folder for visualizations
output_dir = BASE_DIR / "assets"
output_dir.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# Chart 1: Distribution of final grades
plt.figure(figsize=(9, 5))
sns.histplot(
    data=df,
    x="G3",
    bins=21,
    color="royalblue"
)
plt.title("Distribution of Student Final Grades", fontsize=14)
plt.xlabel("Final Grade (0-20)", fontsize=12)
plt.ylabel("Number of Students", fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / "grade_distribution.png", dpi=150)
plt.close()

# Chart 2: Study time and final grades
study_avg = (
    df.groupby("studytime")["G3"]
    .mean()
    .reindex([1, 2, 3, 4])
)
plt.figure(figsize=(9, 5))
plt.bar(
    study_avg.index.astype(str),
    study_avg.values,
    color="steelblue"
)
plt.title("Average Final Grade by Study Time", fontsize=14)
plt.xlabel("Weekly Study Time Category (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)", fontsize=12)
plt.ylabel("Average Final Grade", fontsize=12)
plt.ylim(0, 20)
plt.tight_layout()
plt.savefig(output_dir / "studytime_vs_grade.png", dpi=150)
plt.close()

print("\nCharts saved successfully to assets/ folder!")
