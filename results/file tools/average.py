import os
import pandas as pd

list_data = os.listdir("../../data")
func = lambda string: string.split('.')[0]
list_datasets = list(map(func, list_data))

list_dir = [d for d in os.listdir("../") if os.path.isdir(os.path.join("../", d))]

meta_cols = ['example_id', 'question_index', 'question_polarity',
             'context_condition', 'category', 'answer_info',
             'additional_metadata', 'context', 'question',
             'ans0', 'ans1', 'ans2', 'label']

for dir in list_dir:
    os.chdir("./" + dir)
    list_files = os.listdir()

    groups = {
        d: [s for s in list_files if s.startswith(d)]
        for d in list_datasets
    }

    for dataset_name, files in groups.items():
        if not files:
            continue

        dfs = [pd.read_parquet(f) for f in files]

        # Dynamically detect score_ans columns (model-agnostic)
        score_cols = [c for c in dfs[0].columns if c.startswith('score_ans')]

        # Average only the score columns row-by-row
        avg_scores = pd.concat(
            [df[score_cols] for df in dfs]
        ).groupby(level=0).mean()

        # Combine metadata + averaged scores
        metadata = dfs[0][meta_cols].reset_index(drop=True)
        result = pd.concat([metadata, avg_scores.reset_index(drop=True)], axis=1)

        output_name = f"{dataset_name}-average.parquet"
        result.to_parquet(output_name, index=False)
        print(f"Saved: {output_name} ({len(result)} rows from {len(files)} files)")

    os.chdir("../../")