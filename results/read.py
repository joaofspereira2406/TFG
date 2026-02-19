# inspect_parquet.py
import argparse
import pandas as pd


PARQUET_PATH = "facebook_bart_large_mnli/BBQ-AGE_batched_predictions_20260217_032619.parquet"



pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 200)

df = pd.read_parquet(PARQUET_PATH)

print(f"File: {PARQUET_PATH}")
print(f"Shape: {df.shape}\n")

print("Columns:")
print(list(df.columns), "\n")

print("Dtypes:")
print(df.dtypes, "\n")

print("Head(10):")
print(df.head(10), "\n")

# Quick check of prediction columns (if present)
pred_cols = [c for c in df.columns if c.startswith("pred__")]
score_cols = [c for c in df.columns if c.startswith("score__")]

if pred_cols:
    print("Prediction columns value counts (top 10 each):")
    for c in pred_cols:
        print(f"\n{c}:")
        print(df[c].value_counts(dropna=False).head(10))

if score_cols:
    print("\nScore columns summary:")
    print(df[score_cols].describe().T)

