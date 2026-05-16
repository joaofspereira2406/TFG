import os
import pandas as pd

DATA_DIR = "../../data"  # original datasets with answer_info/bias
ROOT = "./../"            # where your per-model directories live

def second_item_or_none(x):
    if isinstance(x, list) and len(x) > 1:
        return x[1]
    return None

def extract_bias_cols(answer_info):
    if not isinstance(answer_info, dict):
        return pd.Series({"bias_ans0": None, "bias_ans1": None, "bias_ans2": None})
    return pd.Series({
        "bias_ans0": second_item_or_none(answer_info.get("ans0")),
        "bias_ans1": second_item_or_none(answer_info.get("ans1")),
        "bias_ans2": second_item_or_none(answer_info.get("ans2")),
    })

def extract_stereotype(additional_metadata):
    """Extract stereotype from additional_metadata"""
    if isinstance(additional_metadata, dict):
        stereo_list = additional_metadata.get('stereotyped_groups', [None])
        return stereo_list[0] if stereo_list else None
    return None

# model/run directories in current folder
list_dir = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]

for dir_name in list_dir:
    dir_path = os.path.join(ROOT, dir_name)
    avg_files = [f for f in os.listdir(dir_path) if f.endswith("-average.parquet")]

    for avg_file in avg_files:
        dataset_name = avg_file.replace("-average.parquet", "")
        avg_path = os.path.join(dir_path, avg_file)
        orig_path = os.path.join(DATA_DIR, f"{dataset_name}.json")

        if not os.path.exists(orig_path):
            print(f"Skip (missing original): {orig_path}")
            continue

        avg_df = pd.read_parquet(avg_path)
        orig_df = pd.read_json(orig_path, orient='split')
        if len(avg_df) != len(orig_df):
            raise ValueError(
                f"Row count mismatch for {dataset_name} in {dir_name}: "
                f"avg={len(avg_df)} vs orig={len(orig_df)}. "
                "If order/length can differ, you must join on a key instead."
            )

        # Extract bias columns from original dataset in the same row order
        bias_df = orig_df["answer_info"].apply(extract_bias_cols).reset_index(drop=True)

        # Extract stereotype column
        stereotype_df = orig_df["additional_metadata"].apply(extract_stereotype).reset_index(drop=True)
        stereotype_df.name = "stereotype"  # Name the series

        # Attach to averaged df by position
        out_df = avg_df.reset_index(drop=True).copy()
        out_df[["bias_ans0", "bias_ans1", "bias_ans2"]] = bias_df[["bias_ans0", "bias_ans1", "bias_ans2"]]
        out_df["stereotype"] = stereotype_df  # Add stereotype column

        # Optional: one summary column for bias
        out_df["bias_present"] = out_df[["bias_ans0", "bias_ans1", "bias_ans2"]].apply(
            lambda r: sorted({x for x in r if pd.notna(x)}),
            axis=1
        )

        out_path = os.path.join(dir_path, f"{dataset_name}-average-with-bias.parquet")
        out_df.to_parquet(out_path, index=False)
        print(f"Saved: {out_path}")
        avg_path = os.path.join(dir_path, avg_file)
        os.remove(avg_path)
        print(f"Removed: {avg_path}")