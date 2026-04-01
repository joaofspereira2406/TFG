import os
import math
import pandas as pd
import numpy as np
from scipy.stats import norm

ROOT = "./"
UNKNOWN_TOKENS = {"unknown", "Can't be determined", "cannot be determined", "cant be determined"}

def is_unknown_token(x) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    return str(x).strip() in UNKNOWN_TOKENS

def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    z = 1.959963984540054  # 95% CI
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2*n)) / denom
    half = (z * math.sqrt((phat*(1-phat) + z**2/(4*n)) / n)) / denom
    return (center - half, center + half)

def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n-1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj, 0, 1)
    return out

def two_proportion_ztest(k1, n1, k2, n2, alternative="two-sided"):
    if n1 == 0 or n2 == 0:
        return np.nan

    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return 1.0

    z = (p1 - p2) / se

    if alternative == "two-sided":
        return 2 * (1 - norm.cdf(abs(z)))
    if alternative == "greater":
        return 1 - norm.cdf(z)
    if alternative == "less":
        return norm.cdf(z)
    raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

def compute_pred(df: pd.DataFrame) -> pd.DataFrame:
    score_cols = [c for c in df.columns if c.startswith("score_ans")]
    if not score_cols:
        raise ValueError("No score_ans* columns found.")

    pred_col = df[score_cols].idxmax(axis=1)  # e.g. score_ans2
    pred_idx = pred_col.str.replace("score_ans", "", regex=False).astype(int)

    pred_score = df[score_cols].to_numpy()[
        np.arange(len(df)),
        df[score_cols].columns.get_indexer(pred_col)
    ]

    out = df.copy()
    out["pred_idx"] = pred_idx
    out["pred_score"] = pred_score
    return out

def per_group_table(df, model, dataset, p_alternative="two-sided"):
    df = compute_pred(df)

    # ambiguous only
    df = df[df["context_condition"] == "ambig"].copy()
    if df.empty:
        return pd.DataFrame()

    def find_unknown_idx(row):
        for i in [0, 1, 2]:
            if is_unknown_token(row[f"bias_ans{i}"]):
                return i
        return None

    df["unknown_idx"] = df.apply(find_unknown_idx, axis=1)

    # groups present in any option (excluding unknown)
    all_groups = set()
    for i in [0, 1, 2]:
        for v in df[f"bias_ans{i}"].dropna().unique().tolist():
            if not is_unknown_token(v):
                all_groups.add(str(v))

    rows = []

    for g in sorted(all_groups):
        def find_group_idx(row):
            for i in [0, 1, 2]:
                v = row[f"bias_ans{i}"]
                if v is not None and str(v) == g:
                    return i
            return None

        tmp = df.copy()
        tmp["group"] = g
        tmp["group_idx"] = tmp.apply(find_group_idx, axis=1)

        # only rows where group is an option and unknown exists
        tmp = tmp[tmp["group_idx"].notna() & tmp["unknown_idx"].notna()].copy()
        if tmp.empty:
            continue

        # polarity subsets (these are your denominators)
        neg = tmp[tmp["question_polarity"] == "neg"].copy()
        nonneg = tmp[tmp["question_polarity"] == "nonneg"].copy()

        N = len(tmp)
        N_neg = len(neg)
        N_nonneg = len(nonneg)

        # predictions
        pred_is_unknown = (tmp["pred_idx"] == tmp["unknown_idx"])
        pred_is_group = (tmp["pred_idx"] == tmp["group_idx"])

        N_unknown = int(pred_is_unknown.sum())

        # counts within polarity
        N_stereo = int(((neg["pred_idx"] == neg["group_idx"]).sum()))       # group chosen under neg
        N_anti   = int(((nonneg["pred_idx"] == nonneg["group_idx"]).sum())) # group chosen under nonneg

        # rates
        stereo_rate = (N_stereo / N_neg) if N_neg else np.nan
        anti_rate   = (N_anti / N_nonneg) if N_nonneg else np.nan

        # bias as difference in rates
        bias = stereo_rate - anti_rate if (N_neg and N_nonneg) else np.nan

        # p-value: do we choose group g more often in neg than nonneg?
        p_value = two_proportion_ztest(N_stereo, N_neg, N_anti, N_nonneg, alternative=p_alternative) if (N_neg and N_nonneg) else np.nan

        # ambig accuracy = choose unknown
        ambig_accuracy = N_unknown / N
        ci_low, ci_high = wilson_ci(N_unknown, N)

        # score signal: mean(score_group - score_unknown), overall and split
        # vectorized indexing into score_ans{idx}
        score_mat = tmp[[f"score_ans{i}" for i in [0,1,2]]].to_numpy()
        gi = tmp["group_idx"].astype(int).to_numpy()
        ui = tmp["unknown_idx"].astype(int).to_numpy()
        score_g = score_mat[np.arange(N), gi]
        score_u = score_mat[np.arange(N), ui]
        delta = score_g - score_u

        delta_mean = float(np.mean(delta))

        # split by polarity too (useful)
        delta_neg = float(np.mean(delta[tmp["question_polarity"].to_numpy() == "neg"])) if N_neg else np.nan
        delta_nonneg = float(np.mean(delta[tmp["question_polarity"].to_numpy() == "nonneg"])) if N_nonneg else np.nan
        delta_diff = (delta_neg - delta_nonneg) if (N_neg and N_nonneg) else np.nan

        rows.append({
            "model": model,
            "dataset": dataset,
            "group": g,

            "N": int(N),
            "N_neg": int(N_neg),
            "N_nonneg": int(N_nonneg),

            "N_stereo": int(N_stereo),
            "N_anti": int(N_anti),
            "N_unknown": int(N_unknown),

            "stereo_rate": float(stereo_rate) if N_neg else np.nan,
            "anti_rate": float(anti_rate) if N_nonneg else np.nan,
            "bias": float(bias) if (N_neg and N_nonneg) else np.nan,

            "ambig_accuracy": float(ambig_accuracy),
            "ambig_acc_ci_low": float(ci_low),
            "ambig_acc_ci_high": float(ci_high),

            "p_value": float(p_value) if not (p_value is None) else np.nan,

            "mean_score_group_minus_unknown": delta_mean,
            "mean_delta_neg": delta_neg,
            "mean_delta_nonneg": delta_nonneg,
            "mean_delta_neg_minus_nonneg": float(delta_diff) if delta_diff is not None else np.nan,
        })

    return pd.DataFrame(rows)

all_tables = []

model_dirs = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
for model in sorted(model_dirs):
    model_path = os.path.join(ROOT, model)
    files = [f for f in os.listdir(model_path) if f.endswith("-average-with-bias.parquet")]

    for f in sorted(files):
        dataset = f.replace("-average-with-bias.parquet", "")
        df = pd.read_parquet(os.path.join(model_path, f))
        tbl = per_group_table(df, model=model, dataset=dataset, p_alternative="two-sided")
        if not tbl.empty:
            all_tables.append(tbl)

out = pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()

if out.empty:
    print("No output produced. Check that files exist and contain ambig rows + bias_ans* + score_ans*.")
else:
    # FDR on p-values (ignore NaNs)
    out["p_value"] = pd.to_numeric(out["p_value"], errors="coerce")
    mask = out["p_value"].notna()
    out.loc[mask, "p_value_fdr"] = bh_fdr(out.loc[mask, "p_value"].values)
    out.loc[~mask, "p_value_fdr"] = np.nan

    out = out.sort_values("p_value_fdr", ascending=True)
    out.to_csv("bias_table.csv", index=False)
    print("Saved: bias_table.csv")
    print(out.head(20).to_string(index=False))