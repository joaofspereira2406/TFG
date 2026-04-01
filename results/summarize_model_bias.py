import os
import json
import pandas as pd
import numpy as np

ROOT = "./"  # folder containing your model directories
NET_BIAS_FILENAME = "net_bias.json"

rows = []

model_dirs = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]

for model in sorted(model_dirs):
    model_path = os.path.join(ROOT, model)
    json_path = os.path.join(model_path, NET_BIAS_FILENAME)

    if not os.path.exists(json_path):
        # no net_bias.json for this model dir
        continue

    print("Loading:", json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Each key is a dataset, value has {"per_group": [...], "overall": {...}}
    for dataset_name, payload in data.items():
        per_group = payload.get("per_group", [])
        if not per_group:
            continue

        df = pd.DataFrame(per_group)

        # guard against missing fields
        if "group" not in df.columns or "net_bias" not in df.columns:
            continue

        df["net_bias"] = pd.to_numeric(df["net_bias"], errors="coerce")
        df = df.dropna(subset=["net_bias"])

        if df.empty:
            continue

        # Most negative stereotyping (largest positive net_bias)
        top_neg = df.loc[df["net_bias"].idxmax()]

        # Most positive preference (most negative net_bias)
        top_pos = df.loc[df["net_bias"].idxmin()]

        # Strongest effect either direction (largest absolute net_bias)
        top_abs = df.loc[df["net_bias"].abs().idxmax()]

        rows.append({
            "model": model,
            "dataset": dataset_name,

            "top_negative_net_bias_group": top_neg["group"],
            "top_negative_net_bias": float(top_neg["net_bias"]),

            "top_positive_net_bias_group": top_pos["group"],
            "top_positive_net_bias": float(top_pos["net_bias"]),

            "top_abs_net_bias_group": top_abs["group"],
            "top_abs_net_bias": float(top_abs["net_bias"]),
        })

summary = pd.DataFrame(rows)
"""

if summary.empty:
    print("No summaries produced. Check that net_bias.json exists and has per_group data.")
else:
    # Optional: sort so the strongest abs bias per dataset/model shows first
    summary = summary.sort_values(["model", "dataset"])

    print(summary.to_string(index=False))
    summary.to_csv("net_bias_summary.csv", index=False)
    print("\nSaved: net_bias_summary.csv")
"""
# pick the single strongest absolute net bias across ALL datasets for each model
summary["abs_top_abs_net_bias"] = summary["top_abs_net_bias"].abs()
idx = summary.groupby("model")["abs_top_abs_net_bias"].idxmax()
one_line = summary.loc[idx].drop(columns=["abs_top_abs_net_bias"]).sort_values("model")
one_line.to_csv("net_bias_top_per_model.csv", index=False)
print("Saved: net_bias_top_per_model.csv")