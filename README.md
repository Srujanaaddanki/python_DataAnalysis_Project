# python_DataAnalysis_Project

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

# 1. Import the dataset
df = pd.read_excel("2025-3-21-iolp-buildings (1).xlsx")

# 2. Info about the dataset
print(df.info())
print(df.isnull().sum())

# 3. Handling missing data
df_clean = df.dropna()  # Option: df.fillna(method='ffill') or df.fillna(value)
print(df_clean.isnull().sum())

# 4. Statistical Summary
print(df_clean.describe())

# 5. Correlation and relationships
print(df_clean.corr(numeric_only=True))

# 6. Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_clean.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Heatmap of Correlations")
plt.show()

# 7. Outlier Detection using Boxplot and Z-Score
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

# Boxplot (without for loop)
plt.figure(figsize=(16, 12))
for idx, col in enumerate(numeric_cols[:6]):  # limit to 6 for display
    plt.subplot(3, 2, idx + 1)
    sns.boxplot(x=df_clean[col], color='lightblue')
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Z-Score based outlier removal (optional)
z_scores = np.abs(zscore(df_clean[numeric_cols]))
df_zscore = df_clean[(z_scores < 3).all(axis=1)]
print(f"\nRemoved outliers: {len(df_clean) - len(df_zscore)} rows")

# === Visualization Section ===

# Line Chart
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_zscore[numeric_cols])
plt.title("Line Chart of Numeric Features")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()

# Bar Graph (Mean of columns)
mean_values = df_zscore[numeric_cols].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
mean_values.plot(kind='bar', color='skyblue')
plt.title("Bar Chart: Mean of Numeric Features")
plt.ylabel("Mean Value")
plt.show()

# Column Chart (Same as bar but horizontal)
mean_values.plot(kind='barh', color='lightgreen')
plt.title("Column Chart: Mean of Numeric Features")
plt.xlabel("Mean Value")
plt.show()

# Scatter Plot
if len(numeric_cols) >= 2:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df_zscore[numeric_cols[0]], y=df_zscore[numeric_cols[1]])
    plt.title(f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}")
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.show()

# Pair Plot
sns.pairplot(df_zscore[numeric_cols[:5]])  # limit to 5 columns for readability
plt.suptitle("Pair Plot of Numeric Variables", y=1.02)
plt.show()

# Predictive Analysis (Optional - Linear Regression Example)
from sklearn.linear_model import LinearRegression
