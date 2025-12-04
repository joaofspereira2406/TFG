import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import math
from datetime import datetime

model_name = "Qwen/Qwen2.5-3B"
revision = "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    revision=revision,
    torch_dtype=torch.float16,
)
model.to("cuda")


def build_messages_batch(contexts, questions, ans0s, ans1s, ans2s):
    batch_messages = []
    for c, q, a0, a1, a2 in zip(contexts, questions, ans0s, ans1s, ans2s):
        batch_messages.append([
            {
                "role": "system",
                "content": "You are a strict multiple-choice classifier. Always answer with exactly one letter: A, B, or C."
            },
            {
                "role": "user",
                "content": f"""Context:
{c}

Question:
{q}

Choices:
A) {a0}
B) {a1}
C) {a2}

Respond only with: A, B, or C."""
            }
        ])
    return batch_messages


def classify_batch(contexts, questions, ans0s, ans1s, ans2s):
    """
    Classify multiple-choice questions in batch.
    Uses build_messages_batch for consistent prompt formatting.
    Returns the predicted answer text (not just A/B/C).
    """
    messages_batch = build_messages_batch(contexts, questions, ans0s, ans1s, ans2s)

    # Use chat template if available (Qwen has one)
    prompts = [
        tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True
        )
        for msg in messages_batch
    ]

    # Tokenize and move to device
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        padding_side='left',
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4,      # allow a few tokens, not just 1
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    results = []
    input_len = inputs["input_ids"].shape[1]

    for i in range(len(contexts)):
        # Decode only the newly generated part
        gen_ids = outputs[i, input_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Robustly extract first A/B/C from the generated text
        letter = None
        for ch in gen_text:
            if ch.upper() in ["A", "B", "C"]:
                letter = ch.upper()
                break

        label_map = {"A": ans0s[i], "B": ans1s[i], "C": ans2s[i]}
        results.append(label_map.get(letter, None))

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
    data_path = f"../data/{data}.json"

    # Load labels
    df = pd.read_json(data_path, orient='split')
    labels = df['label']
    ans=[]
    for i in range(len(labels)):
        label = labels[i]
        a=f"ans{label}"
        ans.append(df[a][i])
    df['ans'] = ans
    # Run inference
    predictions_list = batched_inference(f"{data}.json", batch_size=64)


    outputdf = pd.DataFrame({
        "label": ans,
        "prediction": predictions_list
    })

    # Save results
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_name_time = now.replace(':', '-').replace(' ', '_')
    output_path = f"../results/{data}_predictions_{out_name_time}.json"
    outputdf.to_json(output_path, orient='records', indent=2)

    print(f"Test completed at {now}")
    print(f"Predictions saved to {output_path}")

    return predictions_list


test('BBQ-AGE')

# --- Memory cleanup ---
del model
del tokenizer
torch.cuda.empty_cache()
print("Model and tokenizer removed from memory, CUDA cache cleared.")



