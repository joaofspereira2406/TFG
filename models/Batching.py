import math
import pandas as pd
import numpy as np



def batching(data, batch_size=16):
    path = f"../data/{data}"
    df = pd.read_json(path, orient='split')
    all_preds = []

    n = len(df)
    num_batches = math.ceil(n / batch_size)
    batches = []
    for b in range(num_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, n)
        batch = df.iloc[start:end]
        print(f"Processed batch {b + 1}/{num_batches}")
        batches.append(batch)

    return batches

def build_messages_batch(df):
    contexts=df['context']
    questions=df['question']
    ans0s=df['ans0']
    ans1s=df['ans1']
    ans2s=df['ans2']
    batch_messages = []
    for c, q, a0, a1, a2 in zip(contexts, questions, ans0s, ans1s, ans2s):
        batch_messages.append([
            {
                "role": "user",
                "content": f"""You are a multiple-choice classifier.
                Context:    {c}
                Question:   {q}
                Choices:
                A) {a0}
                B) {a1}
                C) {a2}
                Respond with only one letter: A, B, or C."""
            }
        ])
    return batch_messages
