import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# ========== LOGGING SETUP ==========
def setup_logging(log_dir: str = "./logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{log_dir}/debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    loggers = {
        'load_all_models': logging.getLogger('load_all_models'),
        'get_unknown_idx': logging.getLogger('get_unknown_idx'),
        'get_target_idx_from_answer_info': logging.getLogger('get_target_idx_from_answer_info'),
        'get_biased_idx': logging.getLogger('get_biased_idx'),
        'compute_metrics': logging.getLogger('compute_metrics'),
        'compute_per_category_bias': logging.getLogger('compute_per_category_bias'),
        'compute_per_group_bias': logging.getLogger('compute_per_group_bias'),
        'build_summaries': logging.getLogger('build_summaries'),
        'precompute_static': logging.getLogger('precompute_static'),
    }
    return loggers

LOGGERS = setup_logging()


# ── Helper Functions ────

def load_all_models(ROOT: str = "./") -> dict:
    logger = LOGGERS['load_all_models']
    logger.info(f"STARTING load_all_models with ROOT='{ROOT}'")

    if not os.path.exists(ROOT):
        logger.error(f"Root directory '{ROOT}' does not exist!")
        return {}

    all_items = os.listdir(ROOT)
    exclude_dirs = ["file tools", "metrics", "logs", "__pycache__"]
    model_dirs = [
        d for d in all_items
        if os.path.isdir(os.path.join(ROOT, d)) and d not in exclude_dirs
    ]
    logger.info(f"Found {len(model_dirs)} model directories: {model_dirs}")

    all_data = {}
    for model_dir in sorted(model_dirs):
        dir_path = os.path.join(ROOT, model_dir)
        files = [f for f in os.listdir(dir_path) if f.endswith("-average-with-bias.parquet")]
        for f in files:
            file_path = os.path.join(dir_path, f)
            key = f"{model_dir}/{f}"
            try:
                df = pd.read_parquet(file_path)
                all_data[key] = df
                logger.info(f" Loaded {key} ({len(df)} rows)")
            except Exception as e:
                logger.error(f" Failed to load {file_path}: {e}")

    logger.info(f"COMPLETED load_all_models: Loaded {len(all_data)} datasets")
    return all_data


def get_unknown_idx(row_bias):
    for i, b in enumerate(row_bias):
        if str(b).lower() == "unknown":
            return i
    return None

def get_non_target_idx_from_target_loc(target_loc: int, unknown_idx: int) -> int:
    """
    Returns the single non-unknown, non-target index among {0,1,2}.
    If it cannot be determined, returns -1.
    """
    candidates = [0, 1, 2]
    remaining = [i for i in candidates if i != unknown_idx and i != target_loc]
    return remaining[0] if len(remaining) == 1 else -1


def get_biased_idx_from_target_loc(polarity: str, target_loc: int, unknown_idx: int) -> int:
    """
    polarity == 'neg' => biased option is target_loc
    polarity == 'pos' => biased option is the other non-unknown option
    Returns -1 if invalid.
    """
    if target_loc < 0 or target_loc > 2:
        return -1
    if unknown_idx < 0 or unknown_idx > 2:
        return -1
    if target_loc == unknown_idx:
        return -1

    non_target_idx = get_non_target_idx_from_target_loc(target_loc, unknown_idx)
    if non_target_idx == -1:
        return -1

    return int(target_loc if polarity == "neg" else non_target_idx)

# ── NEW: Precompute static per-row indices (shared across all models) ────

def precompute_static(df: pd.DataFrame) -> dict:
    logger = LOGGERS["precompute_static"]

    bias_cols = df[["bias_ans0", "bias_ans1", "bias_ans2"]].values
    polarities = df["question_polarity"].values

    # IMPORTANT: target_loc is 0/1/2 (may be NaN). We coerce safely.
    target_loc_raw = df["target_loc"].values

    N = len(df)

    # Unknown index per row (you said this works)
    unknown_idxs_int = np.array(
        [get_unknown_idx(row) if get_unknown_idx(row) is not None else -1 for row in bias_cols],
        dtype=np.int8,
    )

    # Compute biased idx using target_loc + polarity
    biased_idxs = np.empty(N, dtype=np.int8)

    bad_target_loc = 0
    bad_structure = 0

    for i in range(N):
        u = int(unknown_idxs_int[i])
        if u < 0:
            biased_idxs[i] = -1
            continue

        tl = target_loc_raw[i]
        if tl is None or (isinstance(tl, float) and np.isnan(tl)):
            bad_target_loc += 1
            biased_idxs[i] = -1
            continue

        tl = int(tl)

        bi = get_biased_idx_from_target_loc(polarities[i], tl, u)
        if bi == -1:
            bad_structure += 1
        biased_idxs[i] = bi

    valid_mask = (unknown_idxs_int >= 0) & (biased_idxs >= 0)

    logger.info(f"Valid rows: {int(valid_mask.sum())}/{N}")
    if bad_target_loc:
        logger.warning(f"Invalid rows due to missing/NaN target_loc: {bad_target_loc}")
    if bad_structure:
        logger.warning(f"Invalid rows due to inconsistent (target_loc, unknown_idx, polarity): {bad_structure}")

    return {
        "unknown_idxs": unknown_idxs_int,
        "biased_idxs": biased_idxs,
        "valid_mask": valid_mask,
    }


# ── Vectorized metric computation ────

def compute_metrics(df: pd.DataFrame, static: dict) -> dict:
    """Compute dataset-level metrics using precomputed static indices."""
    logger = LOGGERS['compute_metrics']

    scores = df[["score_ans0", "score_ans1", "score_ans2"]].values
    labels = df["label"].values
    context = df["context_condition"].values
    N = len(df)

    unknown_idxs = static["unknown_idxs"]
    biased_idxs = static["biased_idxs"]
    valid_mask = static["valid_mask"]

    pred_labels = np.argmax(scores, axis=1)
    disambig_mask = context == "disambig"
    ambig_mask = context == "ambig"

    # ── Vectorized score extraction ──
    row_idx = np.arange(N)
    # Safe indexing: clamp invalid to 0 (they'll be excluded by valid_mask)
    safe_unk = np.where(unknown_idxs >= 0, unknown_idxs, 0)
    safe_bias = np.where(biased_idxs >= 0, biased_idxs, 0)

    p_unknown = scores[row_idx, safe_unk]
    p_biased = scores[row_idx, safe_bias]
    p_nonunk = 1.0 - p_unknown

    # ── Soft disambig bias ──
    dis_mask = disambig_mask & valid_mask & (p_nonunk > 0)
    soft_disambig_bias = (
        2.0 * (p_biased[dis_mask].sum() / p_nonunk[dis_mask].sum()) - 1.0
    ) if dis_mask.any() else np.nan

    # ── Soft ambig bias ──
    amb_mask = ambig_mask & valid_mask & (p_nonunk > 0)
    soft_ambig_acc = float(p_unknown[ambig_mask & (unknown_idxs >= 0)].mean()) if ambig_mask.any() else np.nan
    if amb_mask.any():
        inner = 2.0 * (p_biased[amb_mask].sum() / p_nonunk[amb_mask].sum()) - 1.0
        soft_ambig_bias = (1.0 - soft_ambig_acc) * inner
    else:
        soft_ambig_bias = np.nan

    # ── Hard metrics ──
    hard_accuracy = float(np.mean(pred_labels == labels))
    hard_disambig_accuracy = float(np.mean(pred_labels[disambig_mask] == labels[disambig_mask])) if disambig_mask.any() else np.nan
    hard_ambig_accuracy = float(np.mean(pred_labels[ambig_mask] == labels[ambig_mask])) if ambig_mask.any() else np.nan

    logger.info(f"soft_disambig_bias={soft_disambig_bias:.4f}, soft_ambig_bias={soft_ambig_bias:.4f}, hard_acc={hard_accuracy:.4f}")

    return {
        "soft_disambig_bias": soft_disambig_bias,
        "soft_ambig_accuracy": soft_ambig_acc,
        "soft_ambig_bias": soft_ambig_bias,
        "hard_accuracy": hard_accuracy,
        "hard_disambig_accuracy": hard_disambig_accuracy,
        "hard_ambig_accuracy": hard_ambig_accuracy,
    }


def compute_per_category_bias(df: pd.DataFrame, static: dict) -> pd.DataFrame:
    """Vectorized per-category metrics using precomputed static indices."""
    logger = LOGGERS['compute_per_category_bias']

    scores = df[["score_ans0", "score_ans1", "score_ans2"]].values
    labels = df["label"].values
    context = df["context_condition"].values
    N = len(df)

    unknown_idxs = static["unknown_idxs"]
    biased_idxs = static["biased_idxs"]
    valid_mask = static["valid_mask"]

    row_idx = np.arange(N)
    safe_unk = np.where(unknown_idxs >= 0, unknown_idxs, 0)
    safe_bias = np.where(biased_idxs >= 0, biased_idxs, 0)

    p_unknown = scores[row_idx, safe_unk]
    p_biased = scores[row_idx, safe_bias]
    p_nonunk = 1.0 - p_unknown
    pred_labels = np.argmax(scores, axis=1)

    categories = df["category"].values
    results = {}

    for cat in np.unique(categories):
        cat_mask = categories == cat
        disambig_mask = (context == "disambig") & cat_mask
        ambig_mask = (context == "ambig") & cat_mask

        # soft_disambig_bias
        dis_mask = disambig_mask & valid_mask & (p_nonunk > 0)
        soft_disambig_bias = (
            2.0 * (p_biased[dis_mask].sum() / p_nonunk[dis_mask].sum()) - 1.0
        ) if dis_mask.any() else np.nan

        # soft_ambig_bias
        amb_mask = ambig_mask & valid_mask & (p_nonunk > 0)
        amb_unk_mask = ambig_mask & (unknown_idxs >= 0)
        soft_ambig_acc = float(p_unknown[amb_unk_mask].mean()) if amb_unk_mask.any() else np.nan
        if amb_mask.any():
            inner = 2.0 * (p_biased[amb_mask].sum() / p_nonunk[amb_mask].sum()) - 1.0
            soft_ambig_bias = (1.0 - soft_ambig_acc) * inner if not np.isnan(soft_ambig_acc) else np.nan
        else:
            soft_ambig_bias = np.nan

        # hard accuracies
        hard_disambig_accuracy = float(np.mean(pred_labels[disambig_mask] == labels[disambig_mask])) if disambig_mask.any() else np.nan
        hard_ambig_accuracy = float(np.mean(pred_labels[ambig_mask] == labels[ambig_mask])) if ambig_mask.any() else np.nan

        results[cat] = {
            "soft_disambig_bias": soft_disambig_bias,
            "soft_ambig_accuracy": soft_ambig_acc,
            "soft_ambig_bias": soft_ambig_bias,
            "hard_disambig_accuracy": hard_disambig_accuracy,
            "hard_ambig_accuracy": hard_ambig_accuracy,
        }
        logger.debug(f"Category {cat}: disambig_bias={soft_disambig_bias:.4f}, ambig_bias={soft_ambig_bias:.4f}")

    logger.info(f"Processed {len(results)} categories")
    return pd.DataFrame(results).T


def compute_per_group_bias(df: pd.DataFrame, static: dict, group_col: str, dataset_name: str) -> pd.DataFrame:
    logger = LOGGERS['compute_per_group_bias']
    logger.info(f"Starting compute_per_group_bias for {dataset_name}")
    try:
        # Log DataFrame info
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Group column: {group_col}, unique values: {df[group_col].unique()}")

        # Prepare arrays
        scores = df[["score_ans0", "score_ans1", "score_ans2"]].values
        labels = df["label"].values
        context = df["context_condition"].values
        groups = df[group_col].values
        categories = df["category"].values

        unknown_idxs = static["unknown_idxs"]
        biased_idxs = static["biased_idxs"]
        valid_mask = static["valid_mask"]

        row_idx = np.arange(len(df))
        safe_unk = np.where(unknown_idxs >= 0, unknown_idxs, 0)
        safe_bias = np.where(biased_idxs >= 0, biased_idxs, 0)

        p_unknown = scores[row_idx, safe_unk]
        p_biased = scores[row_idx, safe_bias]
        p_nonunk = 1.0 - p_unknown
        pred_labels = np.argmax(scores, axis=1)

        results = []
        unique_groups = np.unique(groups)
        logger.info(f"Processing {len(unique_groups)} unique groups")

        for group_idx, group in enumerate(unique_groups):
            logger.debug(f"Group {group_idx + 1}/{len(unique_groups)}: {group}")

            try:
                group_mask = (groups == group)
                logger.debug(f"  group_mask shape: {group_mask.shape}, sum: {group_mask.sum()}")

                # Process each context type
                for ctx in ['disambig', 'ambig']:
                    logger.debug(f"  Processing context: {ctx}")

                    # THIS IS LIKELY WHERE THE ERROR IS
                    # Check if you're doing something like:
                    # ctx_mask = (context == ctx) & group_mask  # This is fine
                    # But then later: if ctx_mask:  # THIS CAUSES THE ERROR!

                    ctx_mask = (context == ctx) & group_mask
                    logger.debug(f"    ctx_mask sum: {ctx_mask.sum()}")

                    # Check for ambiguous truth value - this is the dangerous part
                    # If you have ANY line like this, it will fail:
                    # if ctx_mask:  # ❌ WRONG
                    # if not ctx_mask:  # ❌ WRONG
                    # while ctx_mask:  # ❌ WRONG

                    # Instead, you should be checking:
                    if ctx_mask.any():  # ✓ CORRECT
                        logger.debug(f"    Found {ctx_mask.sum()} valid rows for {ctx}")
                        # Compute your metrics here

                        # Example metric computation (adjust to your actual code)
                        valid_ctx = ctx_mask & valid_mask
                        if ctx == 'disambig':
                            valid_for_bias = valid_ctx & (p_nonunk > 0)
                            if valid_for_bias.any():
                                bias_sum = (2.0 * p_biased[valid_for_bias] / p_nonunk[valid_for_bias] - 1.0).sum()
                                s = bias_sum / valid_for_bias.sum()  # or however you compute

                                results.append({
                                    'group': group,
                                    'category': dataset_name,
                                    'metric': 'disambig_bias',
                                    'value': float(s),
                                    'n_examples': valid_for_bias.sum(),
                                    'n_skipped': (~valid_ctx).sum() if valid_ctx.any() else 0
                                })
                        else:  # ambig
                            ambig_valid = ctx_mask & (unknown_idxs >= 0)
                            if ambig_valid.any():
                                soft_acc = float(p_unknown[ambig_valid].mean())
                                results.append({
                                    'group': group,
                                    'category': dataset_name,
                                    'metric': 'ambig_accuracy',
                                    'value': soft_acc,
                                    'n_examples': ambig_valid.sum(),
                                    'n_skipped': 0
                                })
                    else:
                        logger.debug(f"    No valid rows for {ctx}")

            except Exception as e:
                logger.error(f"Error processing group {group}: {e}")
                import traceback
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                continue

        logger.info(f"Completed processing, returning {len(results)} results")
        return pd.DataFrame(results)

    except Exception as e:
        logger.error(f"Fatal error in compute_per_group_bias: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise


# ── Build summary DataFrames ────

def build_summaries(ROOT: str = "./"):
    logger = LOGGERS['build_summaries']
    logger.info("STARTING BUILD_SUMMARIES")

    metrics_dir = "./metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    all_data = load_all_models(ROOT)
    if not all_data:
        logger.error("No data loaded!")
        return None, None, None, None

    rows, cat_list, group_list, per_example_rows = [], [], [], []

    for i, (key, df) in enumerate(all_data.items(), 1):
        model_name = key.split("/")[0]
        dataset_name = key.split("/")[1].replace("-average-with-bias.parquet", "")
        logger.info(f"[{i}/{len(all_data)}] {model_name} / {dataset_name} ({len(df)} rows)")

        # ── Precompute static indices ONCE per dataset ──
        static = precompute_static(df)

        # Dataset-level
        try:
            m = compute_metrics(df, static)
            m["model"] = model_name
            m["dataset"] = dataset_name
            rows.append(m)
        except Exception as e:
            logger.error(f"Dataset metrics failed: {e}")
            continue

        # Category-level
        try:
            cat_bias = compute_per_category_bias(df, static)
            for cat, row in cat_bias.iterrows():
                for metric in ['soft_disambig_bias', 'soft_ambig_bias', 'soft_ambig_accuracy']:
                    if metric in row.index and not pd.isna(row[metric]):
                        cat_list.append({
                            "model": model_name, "dataset": dataset_name,
                            "category": cat, "metric": metric, "value": row[metric]
                        })
        except Exception as e:
            logger.error(f"Category metrics failed: {e}")

        # Group-level
        try:
            tloc = df["target_loc"].astype("Int64").to_numpy()
            ainfo = df["answer_info"].to_numpy()

            def get_bias_group(row):
                ai = row["answer_info"]
                tl = row["target_loc"]

                if isinstance(ai, dict) and tl == tl:
                    entry = ai.get(f"ans{int(tl)}")
                    # Don't check "if entry" - check "if entry is not None" instead
                    if entry is not None and len(entry) > 1:
                        return entry[1]
                return None

            print("Checking answer_info structure:")
            sample = df["answer_info"].iloc[0]
            print(f"Type of answer_info: {type(sample)}")
            print(f"Sample answer_info: {sample}")
            if isinstance(sample, dict):
                for k, v in sample.items():
                    print(f"  {k}: {type(v)} - {v}")

            # Test get_bias_group on first few rows
            for idx in range(min(3, len(df))):
                result = get_bias_group(df.iloc[idx])
                print(f"Row {idx} result: {type(result)} - {result}")

            df["group"] = df.apply(get_bias_group, axis=1)

            print("group value counts (including None):")
            print(df["group"].value_counts(dropna=False))

            print("\nRows where group is None:")
            none_mask = df["group"].isna()
            print(f"  {none_mask.sum()} / {len(df)} rows")

            if none_mask.sum() > 0:
                print("\nSample of None rows:")
                print(df[none_mask][["bias_ans0", "bias_ans1", "bias_ans2", "target_loc", "answer_info"]].head(5))

            # ADD THIS LOGGING:
            logger.info(f"Unique group values (including None): {df['group'].unique()}")
            logger.info(f"Group column dtype: {df['group'].dtype}")

            # Check for None values in group before passing to compute_per_group_bias
            if df['group'].isna().any():
                logger.warning(f"Found {df['group'].isna().sum()} None/NaN groups - these may cause issues")
                # Option 1: Replace None with a string
                df['group'] = df['group'].fillna('unknown')
                logger.info("Replaced None/NaN groups with 'unknown'")

            group_bias = compute_per_group_bias(df, static, "group", dataset_name)

            for _, row in group_bias.iterrows():
                group_list.append({
                    "model": model_name, "dataset": dataset_name,
                    "category": row['category'], "group": row['group'],
                    "metric": row['metric'], "value": row['value'],
                    "n_examples": row['n_examples'], "n_skipped": row['n_skipped']
                })
        except Exception as e:
            logger.error(f"Group metrics failed: {e}")



    # Assemble & save
    summary_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    cat_df_long = pd.DataFrame(cat_list) if cat_list else pd.DataFrame()
    group_df_long = pd.DataFrame(group_list) if group_list else pd.DataFrame()

    for df_out, path, name in [
        (summary_df,     "./metrics/summary.parquet",          "summary"),
        (cat_df_long,    "./metrics/cat_df.parquet",           "category"),
        (group_df_long,  "./metrics/group_df.parquet",         "group"),

    ]:
        try:
            df_out.to_parquet(path, index=False)
            logger.info(f" Saved {name} -> {path} ({len(df_out)} rows)")
        except Exception as e:
            logger.error(f" Failed to save {name}: {e}")

    print("=" * 90)
    print(f"summary.parquet:          {len(summary_df)} rows")
    print(f"cat_df.parquet:           {len(cat_df_long)} rows")
    print(f"group_df.parquet:         {len(group_df_long)} rows")

    print("=" * 90)

    logger.info("BUILD_SUMMARIES COMPLETED")
    return summary_df, cat_df_long, group_df_long

if __name__ == "__main__":
    main_logger = logging.getLogger('__main__')
    main_logger.info(f"CWD: {os.getcwd()}")

    summary, cat_df, group_df, per_example = build_summaries("./")
