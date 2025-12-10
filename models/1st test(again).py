from torch.cuda import cudart
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch
import pandas as pd
import math
from datetime import datetime
import json

metadata_rows = []
models= {"Qwen/Qwen2.5-3B-Instruct":"aa8e72537993ba99e69dfaafa59ed015b17504d1"}



model_list=models.keys()
for model in model_list:
    model_name=model
    revision=models[model]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained( model_name, revision=revision, dtype=torch.float16, )
    model.to("cuda")
    data='BBQ-AGE'
    name = f"{data}.json"
    path = f"../data/{name}"
    df = pd.read_json(path, orient="split")
    preds=[]
    preds_polarity=[]
    labels_polarity=[]
    dataframes_list=[]


    for i in range(len(df)):
        c = df['context'][i]
        q = df['question'][i]
        a0 = df['ans0'][i]
        a1 = df['ans1'][i]
        a2 = df['ans2'][i]
        ans_info = df['answer_info'][i]
        prompt = f""" 
        System: You are a strict multiple-choice classifier. 
            Always answer with exactly one letter: A, B, or C. 
        User: 
            Context: 
            {c} 
            Question: 
            {q} 
            Choices: A) {a0} B) {a1} C) {a2} \
            Respond only with: A, B, or C. """
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")


        class StopOnNewline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                # decode the last token and check for newline
                decoded = tokenizer.decode(input_ids[0, -1:], skip_special_tokens=False)
                return "\n" in decoded


        stopping_criteria = StoppingCriteriaList([StopOnNewline()])
        # Generate response
        with torch.no_grad():
            outputs = model.generate(**inputs,
                                     max_new_tokens=1,  # short output
                                     do_sample=False,  # deterministic
                                     pad_token_id=tokenizer.eos_token_id,
                                     stopping_criteria=stopping_criteria)
        # Decode and print
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        answer = response[len(prompt):].strip()
        answer = answer.split(" ")[0]
        label = df['label'][i]
        ans = ord(answer)-65
        ans_number = f"ans{ans}"
        ans_polarity = ans_info[ans_number]
        ans_polarity = ans_polarity[1]
        print(ans)
        print(label)
        label_number = f"ans{label}"
        label_polarity = ans_info[label_number]
        label_polarity = label_polarity[1]
        print (f"""{i+1} de {len(df)}\nprediction {answer} --- target {chr(65+label)}\n########################################""")
        preds.append(ans)
        preds_polarity.append(ans_polarity)
        labels_polarity.append(label_polarity)

    new_df = pd.DataFrame({
        'preds': preds,
        'preds_polarity':preds_polarity,
        'label': df['label'],
        'labels_polarity': labels_polarity
    })
    metadata_rows.append({
        "data": new_df.to_dict(orient="records"),
        "model_name": model_name,
        "date": datetime.now().isoformat(),
        "revision": revision,
    })



metadata = pd.DataFrame(metadata_rows)
metadata.set_index('model_name', inplace=True)

metadata.to_json("output_qwen.json", orient="records")



#######################################################
del model
del tokenizer
torch.cuda.empty_cache()
print("Model and tokenizer removed from memory, CUDA cache cleared.")