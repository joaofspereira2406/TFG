import pandas as pd


sets= [
    "BBQ-AGE",
    "BBQ-DISABILITY",
    "BBQ-GENDER",
    "BBQ-NATIONALITY",
    "BBQ-PHYSICAL_APPEARANCE",
    "BBQ-RACE_ETHNICITY",
    "BBQ-RACE_SES",
    "BBQ-RACE_GENDER",
    "BBQ_RELIGION",
    "BBQ_SES",
    "BBQ_SEXUAL_ORIENTATION"
]


for s in sets:
    df = pd.read_json(f"{s}.json", orient='split')
    for i in range(5):
        print(df['context'][i])



