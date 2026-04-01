import os
import re
import pandas as pd

ROOT = "./"

list_dir = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]

for dir_name in list_dir:
    dir_path = os.path.join(ROOT, dir_name)
    parquet_files = [f for f in os.listdir(dir_path) if f.endswith(".parquet")]

    for f in parquet_files:
        path = os.path.join(dir_path, f)
        df = pd.read_parquet(path)

        # Build rename mapping: score_ans0__whatever -> score_ans0
        rename_map = {
            col: re.sub(r'(score_ans\d+).*', r'\1', col)
            for col in df.columns
            if re.match(r'score_ans\d+', col)
        }

        if rename_map:
            df = df.rename(columns=rename_map)
            df.to_parquet(path, index=False)
            print(f"Renamed in {f}: {rename_map}")
        else:
            print(f"No score_ans columns to rename in {f}")