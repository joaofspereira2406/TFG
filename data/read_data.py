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
    ans_info = df["answer_info"]
    keys = ["ans0", "ans1", "ans2"]

    for k in keys:
        second_items = (
            df["answer_info"]
            .dropna()
            .apply(lambda d: (d.get(k, [None, None])[1] if len(d.get(k, [])) > 1 else None))
        )
    
        uniq = pd.unique(second_items.dropna())
        print(f"{k} unique 2nd-item values ({len(uniq)}):")
        for v in uniq:
            print(f"  {v}")
    print(df.columns)
    for i in range(5):
        print(df['answer_info'][i])



