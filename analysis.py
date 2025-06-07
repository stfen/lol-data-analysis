import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

"""
Exploratory Data Analysis of League of Legends Matches
Date: 20.05.2025
Goal: Analyze statistics from the first 10 minutes of League of Legends games 
(ranked Diamond I to Master) and identify factors influencing the blue team's victory.
"""

"""
================================
1. Load and inspect data
================================
"""

df = pd.read_csv("data/high_diamond_ranked_10min.csv")

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
print("\nData types:\n", df.dtypes.value_counts())
print("\nMissing values:\n", df.isnull().sum().sort_values(ascending=False).head())
print("\nStatistical summary:\n", df.describe())

# Select numeric columns
df_numeric = df.select_dtypes(include=np.number)

"""
================================
2. Target variable distribution
================================
"""
plt.figure()
sns.countplot(data=df, x='blueWins', palette='coolwarm')
plt.title("Blue Team Win Distribution")
plt.xlabel("Win (1 = Yes)")
plt.ylabel("Number of Games")
plt.tight_layout()
plt.savefig("blueWins_count.png")
plt.show()

"""
================================
3. Correlation analysis
================================
"""
corrs = df_numeric.corr()
blue_corr = corrs["blueWins"].sort_values(ascending=False)

# Top positive correlations
plt.figure(figsize=(10, 8))
sns.barplot(y=blue_corr.index[1:15], x=blue_corr.values[1:15], palette="viridis")
plt.title("Top Positive Correlations with blueWins")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("figures/top_positive_correlations.png")
plt.show()

# Top negative correlations
plt.figure(figsize=(10, 8))
sns.barplot(y=blue_corr.index[-15:], x=blue_corr.values[-15:], palette="rocket")
plt.title("Top Negative Correlations with blueWins")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("figures/top_negative_correlations.png")
plt.show()

# Full heatmap of correlations between team stats
plt.figure(figsize=(16, 12))
sns.heatmap(corrs.loc[df.columns.str.startswith('blue') | df.columns.str.startswith('red')],
            cmap='coolwarm', annot=False, fmt='.2f', center=0)
plt.title("Correlation Heatmap of Team Features")
plt.tight_layout()
plt.savefig("figures/blue_red_corr_heatmap.png")
plt.show()

"""
================================
4. Distribution plots of key features
================================
"""
plt.figure(figsize=(14, 6))

# Distribution of gold difference
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='blueGoldDiff', hue='blueWins', bins=30, kde=True, palette="Spectral")
plt.title("Gold Difference Distribution")
plt.xlabel("blueGoldDiff")

# Distribution of experience difference
plt.subplot(1, 2, 2)
sns.histplot(data=df, x='blueExperienceDiff', hue='blueWins', bins=30, kde=True, palette="Spectral")
plt.title("Experience Difference Distribution")
plt.xlabel("blueExperienceDiff")

plt.tight_layout()
plt.savefig("figures/blueGold_vs_exp_diff.png")
plt.show()

"""
================================
5. Histograms of all numeric features
================================
"""
def classify_variable(col):
    unique_vals = df[col].dropna().unique()
    if df[col].dropna().isin([0, 1]).all():
        return 'binary'
    elif np.issubdtype(df[col].dtype, np.integer) and len(unique_vals) < 10:
        return 'discrete'
    else:
        return 'continuous'

var_types = {col: classify_variable(col) for col in df_numeric.columns}

for col, var_type in var_types.items():
    plt.figure(figsize=(6, 4))
    
    if var_type == 'binary':
        sns.countplot(data=df, x=col, palette='coolwarm')
        
    elif var_type == 'discrete':
        unique_vals = sorted(df[col].dropna().unique())
        sns.countplot(data=df, x=col, palette='crest', order=unique_vals)
        plt.xticks(ticks=range(len(unique_vals)), labels=unique_vals)
        
    else:  # continuous
        sns.histplot(data=df, x=col, kde=True, bins=30, color='skyblue')

    plt.title(f"Distribution of: {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(f"figures/distribution/{col}_{var_type}.png")
    plt.close()

"""
================================
6. Boxplots with statistics and outlier count
================================
"""

def count_outliers(series):
    """Count number of outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return ((series < lower) | (series > upper)).sum()

for col in df_numeric.columns:
    series = df_numeric[col]
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    median = series.median()
    outliers = count_outliers(series)

    title = (
        f"Q1-Q3: [{Q1:.2f}, {Q3:.2f}]\n"
        f"Median: {median:.2f}\n"
        f"Outliers: {outliers}"
    )

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=series, color='lightblue')
    plt.title(title)
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(f"figures/boxplots/{col}.png")
    plt.close()

"""
================================
7. Random Forest model & feature importance
================================
"""

# Features and labels
X = df.drop(columns=['blueWins'])
y = df['blueWins']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_
features = X.columns
feat_importance = pd.Series(importances, index=features).sort_values(ascending=False)

# Plot top 15 features
plt.figure(figsize=(10, 8))
sns.barplot(x=feat_importance[:15], y=feat_importance.index[:15], palette='viridis')
plt.title("Top 15 Important Features (Random Forest)")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("figures/top_feature_importances_rf.png")
plt.show()
