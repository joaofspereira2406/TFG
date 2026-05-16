import pandas as pd
import os


new_data = pd.read_csv('additional_metadata.csv')  # load once, like a sane person
RESULTS_DIR = "./../results"
mapping  = {
    "BBQ-RACE_GENDER": "Race_x_gender",
    "BBQ-RACE_SES": "Race_x_SES",
    "BBQ-AGE": "Age",
    "BBQ-DISABILITY": "Disability_status",
    "BBQ-GENDER": "Gender_identity",
    "BBQ-NATIONALITY": "Nationality",
    "BBQ-PHYSICAL_APPEARANCE": "Physical_appearance",
    "BBQ-RACE_ETHNICITY": "Race_ethnicity",
    "BBQ_RELIGION": "Religion",
    "BBQ_SES": "SES",
    "BBQ_SEXUAL_ORIENTATION": "Sexual_orientation"
}


all_items = os.listdir(RESULTS_DIR)
exclude_dirs = ["file tools", "metrics", "logs", "__pycache__"]
model_dirs = [
    d for d in all_items
    if os.path.isdir(os.path.join(RESULTS_DIR, d)) and d not in exclude_dirs
]
print(f"model_dirs: {list(model_dir for model_dir in model_dirs)}")
all_data = {}
i=0
a=0
n_d_c = new_data.columns.tolist()
suffix="-average-with-bias.parquet"
for model_dir  in sorted(model_dirs):
    i=i+1
    j=0
    dir_path = os.path.join(RESULTS_DIR, model_dir)

    files = [f for f in os.listdir(dir_path) if f.endswith(suffix) ]

    for file in sorted(files):
        j=j+1
        df = pd.read_parquet(os.path.join(dir_path, file))
        columns = df.columns.tolist()
        common=[]
        for col in columns:
            if col in n_d_c:
                common.append(col)
        s= map.replace(suffix, "")
        cat = mapping.get(s)

        if cat is None:
            print(f"Missing mapping for {s}")
            continue

        filtered = new_data[new_data["category"] == cat]
        filtered = filtered.sort_values(by="example_id")
        
        df = df.reset_index(drop=True)
        filtered = filtered.reset_index(drop=True)
        print(filtered.head())
        
        df = df.merge(
            filtered[["example_id", "target_loc"]],
            on="example_id",
            how="left"
        )
        print(df.head())
        out_path = os.path.join(dir_path, f"{file}")
        df.to_parquet(out_path, index=False)
        print(f"Saved: {out_path}")
        print(f"{j}files in {model_dir}")


