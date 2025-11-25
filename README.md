# python_DataAnalysis_Project
This is the Python Project

# Building Performance Analysis (Python)

**Short:** Exploratory Data Analysis and basic predictive modeling on a building assets dataset using Python (Pandas, NumPy, Matplotlib, Seaborn, scikit-learn).

## Project Overview
- **Objective:** Clean dataset, explore key features, detect outliers, visualize relationships, and train a simple linear regression to understand drivers of the target variable.
- **Dataset:** Public/private Excel file containing building-related features (location, square footage, construction date, lat/long, etc.).

## Key Steps
1. Data loading and inspection (`df.info()`, `df.describe()`)
2. Missing value handling and justification for chosen strategy
3. Outlier detection using Z-score and rationale
4. Visualizations (heatmap, boxplots, pairplots, scatter plots)
5. Simple Linear Regression to study relationships between selected features and the target
6. Summary, key findings and recommendations

## How to run
```bash
pip install -r requirements.txt
python analysis_notebook.ipynb  # or open in Jupyter / Colab
