from transformers import pipeline
from functools import lru_cache
from pathlib import Path
import pandas as pd
import gc
from datetime import datetime
import torch

MODEL_NAMES = [
    "facebook/bart-large-mnli",
    "roberta-large-mnli",
    "valhalla/distilbart-mnli-12-3",
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "Jiva/xlm-roberta-large-it-mnli",
    "NbAiLab/nb-bert-base-mnli",
    "Recognai/zeroshot_selectra_small",
    "cross-encoder/nli-MiniLM2-L6-H768",
    "cross-encoder/nli-deberta-base"
]

@lru_cache(maxsize=None)
def get_zs_pipeline(model_name: str):
    return pipeline(
        "zero-shot-classification",
        model=model_name,
        device=0,  # 0 for GPU, -1 for CPU
    )

def predict_batch_with_model(pipeline, sequence_list: list[str],
                    labels: list[str],
                    model_name: str):

    outs = pipeline(
        sequence_list,
        candidate_labels=labels,
        hypothesis_template="The answer is {}."
    )

    results = []
    for out in outs:  # outs is a list of dicts (one per sequence)
        results.append({
            "model": model_name,
            "labels": out["labels"],
            "scores": out["scores"],
            "prediction": out["labels"][0],
            "prediction_score": out["scores"][0],
        })
    return results

SETS = [
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
    "BBQ_SEXUAL_ORIENTATION",
]

DATA_DIR = Path("..") / "data"
OUT_DIR = Path("..") / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def safe_model_colname(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


for model_name in MODEL_NAMES:

    OUT_DIR_MODEL = Path("..") / "results" / safe_model_colname(model_name)
    OUT_DIR_MODEL.mkdir(parents=True, exist_ok=True)

    # load ONE model, reuse for all sets
    zs = get_zs_pipeline(model_name)
    for i in range(10):
        for s in SETS:
            df = pd.read_json(DATA_DIR / f"{s}.json", orient="split")
            batches = [group for _, group in df.groupby(["ans0", "ans1", "ans2"], sort=False)]

            pred_col = f"pred__{safe_model_colname(model_name)}"
            score_col = f"score__{safe_model_colname(model_name)}"

            # NEW: store scores for all answer options
            score_ans0_col = f"score_ans0__{safe_model_colname(model_name)}"
            score_ans1_col = f"score_ans1__{safe_model_colname(model_name)}"
            score_ans2_col = f"score_ans2__{safe_model_colname(model_name)}"

            if pred_col not in df.columns:
                df[pred_col] = None
            if score_col not in df.columns:
                df[score_col] = None
            for c in [score_ans0_col, score_ans1_col, score_ans2_col]:
                if c not in df.columns:
                    df[c] = None

            for batch in batches:
                first = batch.iloc[0]
                labels = [first["ans0"], first["ans1"], first["ans2"]]

                sequence_list = []
                row_ids = []
                for idx, row in batch.iterrows():
                    sequence_list.append(f"{row['context']} Question: {row['question']}")
                    row_ids.append(idx)

                batch_results = predict_batch_with_model(zs, sequence_list, labels, model_name)

                for idx, res in zip(row_ids, batch_results):
                    df.at[idx, pred_col] = res["prediction"]
                    df.at[idx, score_col] = float(res["prediction_score"])

                    # NEW: fill per-answer scores (aligned to ans0/ans1/ans2)
                    label_to_score = dict(zip(res["labels"], res["scores"]))
                    df.at[idx, score_ans0_col] = float(label_to_score.get(first["ans0"]))
                    df.at[idx, score_ans1_col] = float(label_to_score.get(first["ans1"]))
                    df.at[idx, score_ans2_col] = float(label_to_score.get(first["ans2"]))

            run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            df["run_timestamp"] = run_ts

            out_path = OUT_DIR_MODEL / f"{s}_batched_predictions_{run_ts}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_path, index=True)
            print(f"Saved: {out_path}")

    # free model after finishing all sets for this model
    del zs
    gc.collect()
    torch.cuda.empty_cache()