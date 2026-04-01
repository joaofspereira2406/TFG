import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("bias_table.csv")

ALPHA = 0.05
EFFECT = 0.05

df["abs_bias"] = df["bias"].abs()
df["sig_large"] = (df["p_value_fdr"] < ALPHA) & (df["abs_bias"] >= EFFECT)

# ── 1. Model-level summary + tiers ──────────────────────────────────────────
model_summary = (
    df.groupby("model")
      .agg(
          ambig_accuracy_mean=("ambig_accuracy", "mean"),
          ambig_accuracy_median=("ambig_accuracy", "median"),
          sig_large_cells=("sig_large", "sum"),
          abs_bias_mean=("abs_bias", "mean"),
          abs_bias_median=("abs_bias", "median"),
          mean_delta_shift_mean=("mean_delta_neg_minus_nonneg", "mean"),
          mean_delta_shift_abs_mean=("mean_delta_neg_minus_nonneg",
                                     lambda s: np.nanmean(np.abs(s))),
      )
      .reset_index()
)

def assign_tier(row):
    if row["ambig_accuracy_mean"] < 0.10:
        return "Unreliable"
    if row["ambig_accuracy_mean"] >= 0.30 and row["sig_large_cells"] > 0:
        return "Biased but functional"
    if row["ambig_accuracy_mean"] >= 0.70 and row["sig_large_cells"] == 0:
        return "Least biased"
    return "Functional (mixed)"

model_summary["tier"] = model_summary.apply(assign_tier, axis=1)
model_summary = model_summary.sort_values(
    ["tier", "ambig_accuracy_mean"], ascending=[True, False]
)

# ── 2. Heatmap data: worst group per (model, dataset) ───────────────────────
df_worst = (
    df.sort_values("abs_bias", ascending=False)
      .groupby(["model", "dataset"], as_index=False)
      .head(1)
      .copy()
)

heat = df_worst.pivot(index="model", columns="dataset", values="abs_bias")
ann  = df_worst.pivot(index="model", columns="dataset", values="sig_large").fillna(False)
ann_str = ann.map(lambda x: "*" if x else "")

# order rows by tier then accuracy (same order as model_summary)
tier_order = model_summary["model"].tolist()
heat = heat.reindex([m for m in tier_order if m in heat.index])
ann_str = ann_str.reindex(heat.index)

# add tier label to y-axis tick labels
tier_map = model_summary.set_index("model")["tier"].to_dict()
TIER_MARKER = {"Unreliable": "⚠", "Biased but functional": "⚡", "Least biased": "✓", "Functional (mixed)": "~"}
yticklabels = [
    f"{TIER_MARKER.get(tier_map.get(m, ''), '')} {m}"
    for m in heat.index
]

# ── 3. Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 1,
    figsize=(max(14, 1.3 * heat.shape[1]), 10 + 0.45 * heat.shape[0]),
    gridspec_kw={"height_ratios": [3, 1]}
)

# ── 3a. Heatmap ──────────────────────────────────────────────────────────────
ax_heat = axes[0]
sns.heatmap(
    heat,
    cmap="magma",
    linewidths=0.5,
    linecolor="white",
    annot=ann_str,
    fmt="",
    ax=ax_heat,
    cbar_kws={"label": "max |bias| across groups (ambig, neg vs nonneg)"},
    vmin=0, vmax=heat.max().max(),
)
ax_heat.set_title(
    "Bias heatmap — model × category\n"
    "Cell = max |bias| across groups; * = FDR<0.05 & |bias|≥0.05\n"
    "⚠ Unreliable  ⚡ Biased but functional  ✓ Least biased  ~ Functional (mixed)",
    fontsize=11
)
ax_heat.set_xlabel("Category / dataset")
ax_heat.set_ylabel("")
ax_heat.set_yticklabels(yticklabels, rotation=0, fontsize=9)
ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha="right", fontsize=9)

# ── 3b. Tier summary table ───────────────────────────────────────────────────
ax_tbl = axes[1]
ax_tbl.axis("off")

cols_to_show = ["model", "tier", "ambig_accuracy_mean", "abs_bias_mean",
                "sig_large_cells", "mean_delta_shift_abs_mean"]
col_labels   = ["Model", "Tier", "Ambig acc (mean)", "|Bias| mean",
                "Sig cells", "Δ score shift (abs)"]

tbl_data = model_summary[cols_to_show].copy()
tbl_data["ambig_accuracy_mean"]      = tbl_data["ambig_accuracy_mean"].map("{:.3f}".format)
tbl_data["abs_bias_mean"]            = tbl_data["abs_bias_mean"].map("{:.3f}".format)
tbl_data["sig_large_cells"]          = tbl_data["sig_large_cells"].astype(int)
tbl_data["mean_delta_shift_abs_mean"]= tbl_data["mean_delta_shift_abs_mean"].map("{:.3f}".format)

tbl = ax_tbl.table(
    cellText=tbl_data.values,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.auto_set_column_width(col=list(range(len(col_labels))))

# colour rows by tier
TIER_COLORS = {
    "Unreliable":           "#ffcccc",
    "Biased but functional":"#fff0cc",
    "Least biased":         "#ccffcc",
    "Functional (mixed)":   "#e8e8e8",
}
for row_i, tier_val in enumerate(model_summary["tier"].tolist(), start=1):
    for col_i in range(len(col_labels)):
        tbl[row_i, col_i].set_facecolor(TIER_COLORS.get(tier_val, "white"))

ax_tbl.set_title("Model tier summary", fontsize=10, pad=4)

plt.tight_layout()
plt.savefig("bias_heatmap_with_tiers.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved: bias_heatmap_with_tiers.png")

# ── 4. Also save the CSV ─────────────────────────────────────────────────────
model_summary.to_csv("model_tiers.csv", index=False)
print("Saved: model_tiers.csv")