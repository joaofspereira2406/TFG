import pandas as pd
import os

from numpy.testing.print_coercion_tables import print_coercion_table
from pyparsing import common

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
all_model_w_stereotype = {}
dic_file_wrong = {}
for model_dir  in sorted(model_dirs):
    all_file_w_stereotype = {}
    print(f"Processing: {model_dir}")
    i=i+1
    j=0
    dir_path = os.path.join(RESULTS_DIR, model_dir)
    files = [f for f in os.listdir(dir_path) if f.endswith(suffix)]
    for file in sorted(files):

        counter = 0
        print("=" * 15, file, "=" * 15)
        # print(f"Processing: {file}")
        # print(f"processing Model: {i}, File: {j}")

        df = pd.read_parquet(os.path.join(dir_path, file))
        counter = 0
        cat = mapping.get(file)
        s = file.replace(suffix, "")
        cat = mapping.get(s)

        if cat is None:
            print(f"Missing mapping for {s}")
            continue

        filtered = new_data[new_data["category"] == cat]
        filtered = filtered.sort_values(by="example_id")

        dicio_wrong_stereotype = {}
        for i in range(df.shape[0]):
            row = df.iloc[i]
            row_new = filtered[filtered["example_id"] == row["example_id"]]
            if not row_new.empty:
                row_series = row_new.iloc[0]
            if not row_new.empty:
                target_loc_value = row_new['target_loc'].iloc[0]

            unknown_idx = -1

            for j in range(len(row['answer_info'].keys())):
                if row['answer_info'][f'ans{j}'][1] == 'unknown':
                    unknown_idx = j
                # Skip if target_loc_value is NaN
                if pd.notna(target_loc_value):

                    target_loc = int(target_loc_value)

                    # sanity checks (same spirit as your function)
                    if not (0 <= target_loc <= 2) or not (0 <= unknown_idx <= 2) or target_loc == unknown_idx:
                        continue

                    # get non-target index (same logic as your helper)
                    non_target_idx = next(
                        j for j in range(3)
                        if j != target_loc and j != unknown_idx
                    )

                    # apply EXACT rule from function
                    biased_idx = target_loc if row["question_polarity"] == "neg" else non_target_idx

                    # convert index → value
                    ans_stereotype = row["stereotype"]

                    ans_biased = row["answer_info"][f"ans{biased_idx}"][1]

                    ans_target = row["answer_info"][f"ans{target_loc}"][1]

                    # print(row["question_polarity"])
                    # print(row["context"], row["question"])

                    stereotype_target = ans_target if row["question_polarity"] == "neg" else ans_biased

                    if stereotype_target != ans_stereotype:
                        # print(f"Answer info for {file} is wrong")
                        # print(stereotype_target)
                        # print(f"{stereotype_target} -> {ans_stereotype}")
                        if ans_stereotype not in dicio_wrong_stereotype.keys():
                            dicio_wrong_stereotype[ans_stereotype] = []
                        if ans_stereotype not in all_file_w_stereotype.keys():
                            all_file_w_stereotype[ans_stereotype] = []
                        if ans_stereotype not in all_model_w_stereotype.keys():
                            all_model_w_stereotype[ans_stereotype] = []

                        if stereotype_target not in all_file_w_stereotype[ans_stereotype]:
                            all_file_w_stereotype[ans_stereotype].append(stereotype_target)
                        if stereotype_target not in dicio_wrong_stereotype[ans_stereotype]:
                            dicio_wrong_stereotype[ans_stereotype].append(stereotype_target)
                        if stereotype_target not in all_model_w_stereotype[ans_stereotype]:
                            all_model_w_stereotype[ans_stereotype].append(stereotype_target)


                        counter += 1
        dic_file_wrong[file,model_dir] = counter
        print(dicio_wrong_stereotype)

        print(dic_file_wrong)

    # Sort dictionary by length of its value lists (ascending order)
    sorted_asc_dict = dict(sorted(all_file_w_stereotype.items(), key=lambda item: len(item[1])))
    sorted_des_dict = dict(sorted(all_file_w_stereotype.items(), key=lambda item: len(item[1]), reverse=True))
    print(sorted_asc_dict)
    print(sorted_des_dict)


sorted_asc_dict = dict(sorted(all_model_w_stereotype.items(), key=lambda item: len(item[1])))
sorted_des_dict = dict(sorted(all_model_w_stereotype.items(), key=lambda item: len(item[1]), reverse=True))
print(sorted_asc_dict)
print(sorted_des_dict)
print(dic_file_wrong)