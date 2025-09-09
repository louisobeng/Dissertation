import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib import gridspec

# CONFIG 
emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
gender_labels  = ["Man", "Woman"]

# Fairness thresholds
bias_gap_threshold   = 0.10                 # |Acc_man - Acc_woman| > threshold → Bias
f1_gender_good_min   = 0.70                 # target weighted F1 (gender) considered adequate
disp_impact_low      = 0.80                 # acceptable DI band (lower)
disp_impact_high     = 1.25                 # acceptable DI band (upper)
equal_opp_band       = 0.05                 # |EOD| ≤ band considered good

# Keep CKplus last
dataset_order = ["FER2013plus", "AffectNet", "CelebA", "CKplus"]

# Paths
CSV_DIR = "/Users/mct/Documents/GitHub/dissertation/Gender_Prediction/from_scratch"
models_info = [
    # FER2013plus
    ("FER2013plus", "baseline",        f"{CSV_DIR}/FER/multitask_predictions_ph_cbam.csv"),
    ("FER2013plus", "MobileNet_V3",    f"{CSV_DIR}/FER/multitask_Scratch_predictions_mnv3.csv"),
    ("FER2013plus", "EfficientNet_B2", f"{CSV_DIR}/FER/multitask_Scratch_predictions_efficientnetb2.csv"),

    # AffectNet
    ("AffectNet",   "baseline",        f"{CSV_DIR}/AffectNET/multitask_predictions_ph_cbam.csv"),
    ("AffectNet",   "MobileNet_V3",    f"{CSV_DIR}/AffectNET/multitask_Scatch_predictions_mnv3.csv"),
    ("AffectNet",   "EfficientNet_B2", f"{CSV_DIR}/AffectNET/multitask_predictions_Sratch_efficientnetb2.csv"),

    # CelebA
    ("CelebA",      "baseline",        f"{CSV_DIR}/CelebA/multitask_predictions_ph_cbam.csv"),
    ("CelebA",      "MobileNet_V3",    f"{CSV_DIR}/CelebA/multitask_Scatch_predictions_mnv3.csv"),
    ("CelebA",      "EfficientNet_B2", f"{CSV_DIR}/CelebA/multitask_predictions_Sratch_efficientnetb2.csv"),

    # CKplus
    ("CKplus",      "baseline",        f"{CSV_DIR}/CKplus/multitask_predictions_ph_cbam.csv"),
    ("CKplus",      "MobileNet_V3",    f"{CSV_DIR}/CKplus/multitask_Scratch_predictions_mnv3.csv"),
    ("CKplus",      "EfficientNet_B2", f"{CSV_DIR}/CKplus/multitask_predictions_Sratch_efficientnetb2.csv"),
]

#  BUILD LONG TABLE 
rows = []
for dataset, model, path in models_info:
    try:
        if not os.path.exists(path):
            print(f" Missing CSV → {dataset} | {model}: {path}")
            continue

        df = pd.read_csv(path).dropna()

        # Global (per-file) fairness metrics
        f1_gender = f1_score(df["true_gender"], df["pred_gender"], average="weighted")

        pred_dist = df["pred_gender"].value_counts(normalize=True)
        disp_impact = pred_dist.get("Woman", 0) / pred_dist.get("Man", 1e-6)

        correct = df[df["true_emotion"] == df["pred_emotion"]]
        man_den   = max(1, (df["true_gender"] == "Man").sum())
        woman_den = max(1, (df["true_gender"] == "Woman").sum())
        man_opp   = (correct["true_gender"] == "Man").sum() / man_den
        woman_opp = (correct["true_gender"] == "Woman").sum() / woman_den
        equal_opp_diff = woman_opp - man_opp

        # Per-emotion subgroup metrics
        for emo in emotion_labels:
            emo_df = df[df["true_emotion"] == emo]
            mdf = emo_df[emo_df["true_gender"] == "Man"]
            wdf = emo_df[emo_df["true_gender"] == "Woman"]

            man_acc = (mdf["pred_gender"] == "Man").mean() if not mdf.empty else np.nan
            woman_acc = (wdf["pred_gender"] == "Woman").mean() if not wdf.empty else np.nan
            emo_acc = (emo_df["true_emotion"] == emo_df["pred_emotion"]).mean() if not emo_df.empty else np.nan

            if np.isnan(man_acc) or np.isnan(woman_acc):
                acc_gap = np.nan
                bias_flag = "N/A"
            else:
                acc_gap = abs(man_acc - woman_acc)
                bias_flag = "Bias_Detected!" if acc_gap > bias_gap_threshold else "No_Bias"

            rows.append({
                "Dataset": dataset,
                "Model": model,
                "Emotion": emo,
                "Emotion_Acc": emo_acc,
                "Man_Acc": man_acc,
                "Woman_Acc": woman_acc,
                "Acc_Gap": acc_gap,
                "Bias_Flag": bias_flag,
                "F1_Gender": f1_gender,
                "Disp_Impact": disp_impact,
                "Equal_Opp_Diff": equal_opp_diff,
            })

    except Exception as e:
        print(f" {dataset}-{model}: {e}")

final_df = pd.DataFrame(rows)

# SUMMARY CSV (page-independent) 
summary = (
    final_df
    .groupby(["Dataset", "Model"], as_index=False)
    .agg(
        mean_Emotion_Acc=("Emotion_Acc","mean"),
        mean_Man_Acc=("Man_Acc","mean"),
        mean_Woman_Acc=("Woman_Acc","mean"),
        mean_Acc_Gap=("Acc_Gap","mean"),
        F1_Gender=("F1_Gender","mean"),
        Disp_Impact=("Disp_Impact","mean"),
        Equal_Opp_Diff=("Equal_Opp_Diff","mean"),
        Emotions_With_Bias_Pct=("Bias_Flag", lambda s: (s.eq("Bias_Detected!").mean()*100.0))
    )
)

summary["Dataset"] = pd.Categorical(summary["Dataset"], categories=dataset_order, ordered=True)
summary = summary.sort_values(["Dataset","Model"]).reset_index(drop=True)

summary_csv_path = "fairness_summary_by_four_dataset_model.csv"
summary.to_csv(summary_csv_path, index=False)
print(f"Wrote summary CSV → {summary_csv_path}")

# FIGURE HELPERS 
def fig_page1_accgap(final_df):
    """Page 1: Acc_Gap heatmap (|Acc_Man − Acc_Woman|) by Emotion, Dataset|Model rows."""
    pivot = (
        final_df
        .pivot_table(index=["Dataset","Model"], columns="Emotion", values="Acc_Gap", aggfunc="mean")
        .reindex(columns=emotion_labels)
    )
    # Sort index with CKplus last (fix ambiguity by not carrying index)
    tmp = pivot.index.to_frame(index=False)
    tmp["Dataset"] = pd.Categorical(tmp["Dataset"], categories=dataset_order, ordered=True)
    tmp = tmp.sort_values(["Dataset","Model"]).reset_index(drop=True)
    new_index = pd.MultiIndex.from_frame(tmp)
    pivot = pivot.reindex(new_index)

    n_rows, n_cols = pivot.shape
    # Color grid: grey for NaN, green OK, red bias
    colors = np.full((n_rows, n_cols, 3), 0.90)
    val_mask = ~pivot.isna().values
    ok_mask   = (pivot.fillna(np.inf).values <= bias_gap_threshold)
    bias_mask = (pivot.fillna(-np.inf).values > bias_gap_threshold)
    colors[ok_mask & val_mask]   = (0.85, 1.00, 0.85)
    colors[bias_mask & val_mask] = (1.00, 0.85, 0.85)

    labels_y = [f"{d} | {m}" for d, m in pivot.index.tolist()]

    fig_h = max(3.0, 0.50 * n_rows + 4.0)
    fig_w = max(6.0, 0.80 * n_cols + 4.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(colors, aspect="auto")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(emotion_labels, rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels_y)

    # Annotate values
    for i in range(n_rows):
        for j in range(n_cols):
            val = pivot.iat[i, j]
            txt = "—" if pd.isna(val) else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    ax.set_title(f"Accuracy Gap by Emotion (|Acc_Man − Acc_Woman|) • Threshold={bias_gap_threshold:.2f}")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Dataset | Model")

    # Legend
    legend_elems = [
        Patch(facecolor=(1.00, 0.85, 0.85), edgecolor='k', label=f"Gap > {bias_gap_threshold:.2f} (Bias)"),
        Patch(facecolor=(0.85, 1.00, 0.85), edgecolor='k', label=f"Gap ≤ {bias_gap_threshold:.2f}"),
        Patch(facecolor=(0.90, 0.90, 0.90), edgecolor='k', label="Missing"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)

    plt.tight_layout()
    return fig

def fig_page2_f1_di_eod(summary):
    """
    Page 2: Three aligned horizontal bar charts (F1 Gender, Disparate Impact, Equal Opp. Diff).
    Only the LEFT panel carries the Dataset|Model labels to keep page clean.
    Adds a suptitle across the top.
    """
    s = summary.copy()
    s["Dataset"] = pd.Categorical(s["Dataset"], categories=dataset_order, ordered=True)
    s = s.sort_values(["Dataset","Model"]).reset_index(drop=True)

    labels_y = [f"{d} | {m}" for d, m in s[["Dataset","Model"]].to_numpy()]
    v_f1  = s["F1_Gender"].astype(float).to_numpy()
    v_di  = s["Disp_Impact"].astype(float).to_numpy()
    v_eod = s["Equal_Opp_Diff"].astype(float).to_numpy()

    fig = plt.figure(figsize=(14, 0.50 * len(s) + 4))
    gs = gridspec.GridSpec(1, 3, wspace=0.25)

    def horiz(ax, vals, title, labels=None, show_labels=False, good_low=None, good_high=None):
        y = np.arange(len(vals))[::-1]
        cols = []
        for v in vals:
            if np.isnan(v):
                cols.append((0.85, 0.85, 0.85))
            else:
                ok = True
                if good_low  is not None and v < good_low:  ok = False
                if good_high is not None and v > good_high: ok = False
                cols.append((0.70, 0.90, 0.70) if ok else (0.95, 0.70, 0.70))
        ax.barh(y, vals, color=cols, edgecolor='k')
        if show_labels:
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=11)
        ax.grid(axis='x', alpha=0.3)

    # LEFT: with labels
    ax1 = fig.add_subplot(gs[0, 0])
    horiz(ax1, v_f1, "F1 – Gender (weighted)", labels=labels_y, show_labels=True, good_low=f1_gender_good_min)

    # MIDDLE: no labels
    ax2 = fig.add_subplot(gs[0, 1])
    horiz(ax2, v_di, "Disparate Impact (P̂[Woman] / P̂[Man])",
          show_labels=False, good_low=disp_impact_low, good_high=disp_impact_high)

    # RIGHT: no labels
    ax3 = fig.add_subplot(gs[0, 2])
    horiz(ax3, v_eod, "Equal Opportunity Difference (Woman − Man)",
          show_labels=False, good_low=-equal_opp_band, good_high=equal_opp_band)

    # Page 2 inscription / suptitle
    fig.suptitle(
        "Fairness Summary: F1 (Gender), Disparate Impact, and Equal Opportunity Difference\n"
        f"(Targets: F1 ≥ {f1_gender_good_min:.2f}; "
        f"{disp_impact_low:.2f} ≤ DI ≤ {disp_impact_high:.2f}; |EOD| ≤ {equal_opp_band:.2f})",
        y=0.98, fontsize=13
    )

    # Shared vertical label (optional)
    fig.text(0.005, 0.5, "Dataset | Model", va='center', rotation='vertical', fontsize=10)

    # Leave room for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig

#  EXPORT TWO-PAGE PDF
with PdfPages("fairness_dashboard_four_pages.pdf") as pdf:
    pdf.savefig(fig_page1_accgap(final_df)); plt.close()
    pdf.savefig(fig_page2_f1_di_eod(summary)); plt.close()

print("PDF written → fairness_dashboard_four_pages.pdf")
