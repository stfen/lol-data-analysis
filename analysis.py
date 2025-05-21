import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Wczytanie danych
df = pd.read_csv("high_diamond_ranked_10min.csv")  # zmień nazwę jeśli plik nazywa się inaczej

# Tylko zmienne liczbowe
df_numeric = df.select_dtypes(include=['int64', 'float64'])

def count_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((series < lower_bound) | (series > upper_bound)).sum()

# Tworzenie boxplotów z opisanym tytułem
for col in df_numeric.columns:
    series = df_numeric[col]
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    median = series.median()
    outliers = count_outliers(series)

    title = (
        f"Q1-Q3: [{Q1:.2f}, {Q3:.2f}]\n"
        f"Mediana: {median:.2f}\n"
        f"Liczba outliers: {outliers}"
    )

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=series, color='lightblue')
    plt.title(title)
    plt.xlabel(col)
    plt.tight_layout()

# Wyświetlenie podstawowych informacji
print("Liczba wierszy:", df.shape[0])
print("Liczba kolumn:", df.shape[1])
print("\nTypy danych:\n", df.dtypes.value_counts())

# Sprawdzenie brakujących danych
print("\nBraki danych:\n", df.isnull().sum().sort_values(ascending=False).head())

# Podstawowy opis statystyczny
print("\nOpis statystyczny:\n", df.describe())

sns.countplot(data=df, x='blueWins', palette='coolwarm')
plt.title("Rozkład zwycięstw drużyny niebieskiej")
plt.xlabel("Zwycięstwo (1 = tak)")
plt.ylabel("Liczba gier")
plt.show()

corrs = df.corr()
blue_corr = corrs["blueWins"].sort_values(ascending=False)

plt.figure(figsize=(8, 12))
sns.barplot(y=blue_corr.index[1:15], x=blue_corr.values[1:15], palette="viridis")
plt.title("Najmocniejsze dodatnie korelacje z blueWins")
plt.xlabel("Współczynnik korelacji")
plt.ylabel("Atrybut")
plt.show()

plt.figure(figsize=(8, 12))
sns.barplot(y=blue_corr.index[-15:], x=blue_corr.values[-15:], palette="rocket")
plt.title("Najmocniejsze ujemne korelacje z blueWins")
plt.xlabel("Współczynnik korelacji")
plt.ylabel("Atrybut")
plt.show()

features = ['blueKills', 'blueAssists', 'blueTotalGold', 'blueEliteMonsters',
            'blueTotalMinionsKilled', 'blueTowersDestroyed', 'blueGoldDiff']

plt.figure(figsize=(15, 10))
for i, feat in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.boxplot(data=df, x='blueWins', y=feat, palette="Set2")
    plt.title(f'{feat} vs blueWins')
plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 12))
sns.heatmap(df.corr().loc[df.columns.str.startswith('blue') | df.columns.str.startswith('red')],
            cmap='coolwarm', annot=False, fmt='.2f', center=0)
plt.title("Macierz korelacji dla wszystkich cech")
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(data=df, x='blueGoldDiff', hue='blueWins', bins=30, kde=True, palette="Spectral")
plt.title("Rozkład różnicy w złocie")

plt.subplot(1, 2, 2)
sns.histplot(data=df, x='blueExperienceDiff', hue='blueWins', bins=30, kde=True, palette="Spectral")
plt.title("Rozkład różnicy w doświadczeniu")

plt.tight_layout()
plt.show()

# Ustawienia wykresów
sns.set(style="whitegrid")
df_numeric = df.select_dtypes(include=['float64', 'int64'])

# Rozkład zmiennych
for col in df_numeric.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f"Rozkład zmiennej: {col}")
    plt.xlabel(col)
    plt.tight_layout()

# Dane i etykiety
X = df.drop(columns=['blueWins'])  # podmień jeśli target się inaczej nazywa
y = df['blueWins']

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Ważność cech
importances = model.feature_importances_
features = X.columns
feat_importance = pd.Series(importances, index=features).sort_values(ascending=False)

# Wykres ważności cech
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_importance[:15], y=feat_importance.index[:15], palette='viridis')
plt.title("Top 15 najważniejszych cech wpływających na wygraną (Random Forest)")
plt.xlabel("Ważność cechy")
plt.ylabel("Cechy")
plt.tight_layout()
plt.show()