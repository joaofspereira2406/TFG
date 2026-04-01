import os
import json
import pandas as pd
import numpy as np

ROOT = "./"
NET_BIAS_FILENAME = "net_bias.json"
TOP_N = 20

rows = []

model_dirs = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]

for model in sorted(model_dirs):
    model_path = os.path.join(ROOT, model)
    json_path = os.path.join(model_path, NET_BIAS_FILENAME)

    if not os.path.exists(json_path):
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for dataset_name, payload in data.items():
        per_group = payload.get("per_group", [])
        if not per_group:
            continue

        df = pd.DataFrame(per_group)

        # expected fields: group, net_bias (plus maybe counts/rates)
        if "group" not in df.columns or "net_bias" not in df.columns:
            continue

        df["net_bias"] = pd.to_numeric(df["net_bias"], errors="coerce")
        df = df.dropna(subset=["net_bias"]).copy()

        if df.empty:
            continue

        df["abs_net_bias"] = df["net_bias"].abs()
        df["model"] = model
        df["dataset"] = dataset_name

        rows.append(df[["model", "dataset", "group", "net_bias", "abs_net_bias"]])

all_bias = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
    columns=["model", "dataset", "group", "net_bias", "abs_net_bias"]
)

if all_bias.empty:
    print("No net bias data found. Check that net_bias.json files exist and contain per_group rows.")
else:
    top20 = all_bias.sort_values("abs_net_bias", ascending=False).head(TOP_N)

    # print nicely
    print(top20.to_string(index=False))

    top20.to_csv("top20_abs_net_bias.csv", index=False)
    print("\nSaved: top20_abs_net_bias.csv")