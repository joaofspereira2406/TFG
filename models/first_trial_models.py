from torch.cuda import cudart
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import math
from datetime import datetime
import json

model_name = "Qwen/Qwen2.5-3B-Instruct"
revision = "aa8e72537993ba99e69dfaafa59ed015b17504d1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    revision=revision,
    dtype=torch.float16,
)
model.to("cuda")


def build_messages_batch(contexts, questions, ans0s, ans1s, ans2s):
    batch_messages = []
    for c, q, a0, a1, a2 in zip(contexts, questions, ans0s, ans1s, ans2s):
        batch_messages.append([
            {
                "role": "user",
                "content": f"""You are a multiple-choice classifier.

Context:
{c}

Question:
{q}

Choices:
A) {a0}
B) {a1}
C) {a2}

Respond with only one letter: A, B, or C."""
            }
        ])
    return batch_messages


def classify_batch(contexts, questions, ans0s, ans1s, ans2s, max_new_tokens=4):
    messages_batch = build_messages_batch(contexts, questions, ans0s, ans1s, ans2s)

    inputs = tokenizer.apply_chat_template(
        messages_batch,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    results = []
    for i in range(len(contexts)):
        # length of non-padding part of input_ids
        input_len_i = (inputs["input_ids"][i] != tokenizer.pad_token_id).sum()
        gen_ids = outputs[i][input_len_i:]
        raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        first_char = raw_output[0].upper() if raw_output else ""
        label_map = {"A": ans0s[i], "B": ans1s[i], "C": ans2s[i]}
        results.append(label_map.get(first_char))
    return results


def batched_inference(data, batch_size=16):
    path = f"../data/{data}"
    df = pd.read_json(path, orient='split')
    all_preds = []

    n = len(df)
    num_batches = math.ceil(n / batch_size)

    for b in range(num_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, n)
        batch = df.iloc[start:end]

        preds = classify_batch(
            batch["context"].tolist(),
            batch["question"].tolist(),
            batch["ans0"].tolist(),
            batch["ans1"].tolist(),
            batch["ans2"].tolist(),
        )
        all_preds.extend(preds)

        print(f"Processed batch {b + 1}/{num_batches}")

    return all_preds


def test(data):
    name = f"{data}.json"

    predictions_list = batched_inference(name, batch_size=64)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_name_time = now.replace(':', '-').replace(' ', '_')

    output_path = f"../results/{data}_predictions_{out_name_time}.json"

    with open(output_path, 'w') as f:
        json.dump(predictions_list, f, indent=2)

    print(f"Test completed at {now}")
    print(f"Predictions saved to {output_path}")

    return predictions_list

test('BBQ-AGE')


# --- Memory cleanup ---
del model
del tokenizer
torch.cuda.empty_cache()
print("Model and tokenizer removed from memory, CUDA cache cleared.")




