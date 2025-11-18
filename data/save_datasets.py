from datetime import datetime

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

text = f"Data last refreshed {now}"

with open("Data_date.txt", "w") as file:
    file.write(text)

###

import pandas as pd

datasets={
    ###BBQ_DATASETS
    "BBQ-AGE":"https://raw.githubusercontent.com/i-gallegos/Fair-LLM-Benchmark/refs/heads/main/BBQ/data/Age.jsonl",
    "BBQ-DISABILITY":"https://raw.githubusercontent.com/i-gallegos/Fair-LLM-Benchmark/refs/heads/main/BBQ/data/Disability_status.jsonl",
    "BBQ-GENDER":"https://raw.githubusercontent.com/i-gallegos/Fair-LLM-Benchmark/refs/heads/main/BBQ/data/Gender_identity.jsonl",
    "BBQ-NATIONALITY":"https://raw.githubusercontent.com/i-gallegos/Fair-LLM-Benchmark/refs/heads/main/BBQ/data/Nationality.jsonl",
    "BBQ-PHYSICAL_APPEARANCE":"https://raw.githubusercontent.com/i-gallegos/Fair-LLM-Benchmark/refs/heads/main/BBQ/data/Physical_appearance.jsonl",
    "BBQ-RACE_ETHNICITY":"https://raw.githubusercontent.com/i-gallegos/Fair-LLM-Benchmark/refs/heads/main/BBQ/data/Race_ethnicity.jsonl",
    "BBQ-RACE_SES":"https://raw.githubusercontent.com/i-gallegos/Fair-LLM-Benchmark/refs/heads/main/BBQ/data/Race_x_SES.jsonl",
    "BBQ-RACE_GENDER":"https://raw.githubusercontent.com/i-gallegos/Fair-LLM-Benchmark/refs/heads/main/BBQ/data/Race_x_gender.jsonl",
    "BBQ_RELIGION":"https://raw.githubusercontent.com/i-gallegos/Fair-LLM-Benchmark/refs/heads/main/BBQ/data/Religion.jsonl",
    "BBQ_SES":"https://raw.githubusercontent.com/i-gallegos/Fair-LLM-Benchmark/refs/heads/main/BBQ/data/SES.jsonl",
    "BBQ_SEXUAL_ORIENTATION":"https://raw.githubusercontent.com/i-gallegos/Fair-LLM-Benchmark/refs/heads/main/BBQ/data/Sexual_orientation.jsonl"
}


sets=datasets.keys()

for set in sets:
    df = pd.read_json(datasets[set], lines=True)
    name=f"{set}.json"
    df.to_json(name, orient="split")


