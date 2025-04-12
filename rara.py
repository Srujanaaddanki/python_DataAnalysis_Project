import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

df = pd.read_excel("C:\\Users\\hp\\Downloads\\2025-3-21-iolp-buildings (1).xlsx")
print(df)
print(df.head(5))
print(df.tail(5))

print(df.info())
print(df.isnull().sum())


df_clean = df.dropna()  
print(df_clean.isnull().sum())


print(df_clean.describe())


print(df_clean.corr(numeric_only=True))


plt.figure(figsize=(12, 8))
sns.heatmap(df_clean.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Heatmap of Correlations")
plt.show()


numeric_cols = df_clean.select_dtypes(include=[np.number]).columns


plt.figure(figsize=(16, 12))
for idx, col in enumerate(numeric_cols[:6]):  # limit to 6 for display
    plt.subplot(3, 2, idx + 1)
    sns.boxplot(x=df_clean[col], color='lightblue')
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


z_scores = np.abs(zscore(df_clean[numeric_cols]))
df_zscore = df_clean[(z_scores < 3).all(axis=1)]
print(f"\nRemoved outliers: {len(df_clean) - len(df_zscore)} rows")




plt.figure(figsize=(10, 6))
sns.lineplot(data=df_zscore[numeric_cols])
plt.title("Line Chart of Numeric Features")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()


mean_values = df_zscore[numeric_cols].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
mean_values.plot(kind='bar', color='skyblue')
plt.title("Bar Chart: Mean of Numeric Features")
plt.ylabel("Mean Value")
plt.show()


mean_values.plot(kind='barh', color='lightgreen')
plt.title("Column Chart: Mean of Numeric Features")
plt.xlabel("Mean Value")
plt.show()


if len(numeric_cols) >= 2:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df_zscore[numeric_cols[0]], y=df_zscore[numeric_cols[1]])
    plt.title(f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}")
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.show()


sns.pairplot(df_zscore[numeric_cols[:5]])  
plt.suptitle("Pair Plot of Numeric Variables", y=1.02)
plt.show()


from sklearn.linear_model import LinearRegression


if len(numeric_cols) >= 3:
    X = df_zscore[[numeric_cols[0], numeric_cols[1]]]
    y = df_zscore[numeric_cols[-1]]
    model = LinearRegression()
    model.fit(X, y)
    print("\n--- Linear Regression Coefficients ---")
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)
