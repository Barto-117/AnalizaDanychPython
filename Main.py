import pandas as pd
import numpy as np


df = pd.read_csv("Diet_R.csv")

df.columns = df.columns.str.strip()

if df["gender"].dtype in [np.int64, np.float64] or set(df["gender"].unique()) == {0, 1}:
    df["gender"] = df["gender"].map({0: "Male", 1: "Female"})

for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (data < lower) | (data > upper)

outliers = df.select_dtypes(include=[np.number]).apply(detect_outliers_iqr)
print("Liczba wartości odstających w każdej kolumnie:")
print(outliers.sum())

def basic_stats(data):
    return pd.Series({
        "Średnia": data.mean(),
        "Mediana": data.median(),
        "Odch.std": data.std(),
        "1 kwartyl": data.quantile(0.25),
        "3 kwartyl": data.quantile(0.75)
    })


stats_all = df.select_dtypes(include=[np.number]).apply(basic_stats)

stats_by_gender = df.groupby("gender").apply(
    lambda x: x.select_dtypes(include=[np.number]).apply(basic_stats)
)

stats_all.to_csv("statystyki_calkowite.csv", encoding="utf-8")
stats_by_gender.to_csv("statystyki_plec.csv", encoding="utf-8")

print("\nStatystyki zapisano do plików:")
print(" - statystyki_calkowite.csv")
print(" - statystyki_plec.csv")