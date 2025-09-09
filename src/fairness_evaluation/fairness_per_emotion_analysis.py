import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  CONFIG 
predictions_csv = "/Users/mct/Desktop/Dataset/results/test_data/New_results/MobileNet/multitask_Scratch_predictions_mnv3.csv"
gender_labels = ["Man", "Woman"]
emotion_labels = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear"]
bias_gap_threshold = 0.10

#  LOAD PREDICTIONS 
pred_df = pd.read_csv(predictions_csv)

#  BIAS + FAIRNESS METRICS 
bias_stats = []

for emotion in emotion_labels:
    # Filter only this emotion
    emo_df = pred_df[pred_df["true_emotion"] == emotion]
    gender_acc = {}
    recall_per_gender = {}
    parity_per_gender = {}

    for gender in gender_labels:
        # Subset rows for this gender within the emotion class
        gdf = emo_df[emo_df["true_gender"] == gender]
        n_true = len(gdf)
        if n_true == 0:
            # No samples for this gender → cannot compute
            gender_acc[gender] = np.nan
            recall_per_gender[gender] = np.nan
            parity_per_gender[gender] = np.nan
        else:
            correct = (gdf["pred_gender"] == gender).sum()
            # Accuracy for this gender within this emotion
            gender_acc[gender] = correct / n_true
            # Recall is same as gender accuracy for gender classification
            recall_per_gender[gender] = gender_acc[gender]
            #  Parity: P(pred=gender | true_gender=gender, emotion)
            parity_per_gender[gender] = correct / n_true

    # Accuracy gap between genders for this emotion
    gap = abs(
        (gender_acc.get("Man", 0) if not np.isnan(gender_acc.get("Man", np.nan)) else 0) -
        (gender_acc.get("Woman", 0) if not np.isnan(gender_acc.get("Woman", np.nan)) else 0)
    )

    bias_stats.append({
        "emotion": emotion,
        "Man_acc": gender_acc.get("Man", np.nan),
        "Woman_acc": gender_acc.get("Woman", np.nan),
        "accuracy_gap": gap,
        "bias_flag": "Bias" if gap > bias_gap_threshold else "",
        "Man_parity": parity_per_gender.get("Man", np.nan),
        "Woman_parity": parity_per_gender.get("Woman", np.nan),
        "Man_recall": recall_per_gender.get("Man", np.nan),
        "Woman_recall": recall_per_gender.get("Woman", np.nan),
    })

#SAVE RESULTS 
bias_df = pd.DataFrame(bias_stats)
bias_df.to_csv("gender_bias_per_emotion.csv", index=False)
bias_df.to_csv("fairness_metrics_per_emotion.csv", index=False)

#  PLOT: ACCURACY PER GENDER 
plt.figure(figsize=(10, 6))
x = np.arange(len(bias_df))
bar_width = 0.35
plt.bar(x - bar_width/2, bias_df["Man_acc"], bar_width, label="Man")
plt.bar(x + bar_width/2, bias_df["Woman_acc"], bar_width, label="Woman")
plt.xticks(x, bias_df["emotion"], rotation=45)
plt.ylabel("Gender Prediction Accuracy")
plt.title("Gender Prediction Accuracy per Emotion Class")
plt.legend()
plt.tight_layout()
plt.savefig("gender_bias_per_emotion_plot.png")
plt.show()

#  HEATMAP: FAIRNESS METRICS 
plt.figure(figsize=(12, 6))
sns.heatmap(
    bias_df.set_index("emotion")[[
        "Man_parity", "Woman_parity", "Man_recall", "Woman_recall"
    ]],
    annot=True, cmap="coolwarm", fmt=".2f"
)
plt.title("Fairness Metrics per Emotion")
plt.tight_layout()
plt.savefig("fairness_metrics_per_emotion_heatmap.png")
plt.show()

#  TABLE PREVIEW PNG 
def df_to_img(df, filename):
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.5 + 1))
    ax.axis('off')
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

df_to_img(bias_df, "gender_bias_per_emotion_table.png")
df_to_img(
    bias_df.drop(columns=["bias_flag"]).round(2),
    "fairness_metrics_per_emotion_table.png"
)

# FAIRNESS AUDIT: Gender × Emotion 
def fairness_audit(df, output_path="fairness_report.txt"):
    print("\n Accuracy by Gender × Emotion:")
    # Average emotion classification accuracy by true_gender
    accuracy_table = (
        df.groupby(["true_gender", "true_emotion"])
          .apply(lambda g: (g["true_emotion"] == g["pred_emotion"]).mean())
          .unstack()
          .fillna(0)
    )
    print(accuracy_table.round(3))

    # Disparate Impact (overall gender prediction rates) 
    pos_rate = df["pred_gender"].value_counts(normalize=True)
    disp_impact = pos_rate.get("Woman", 0) / (pos_rate.get("Man", 1e-6))
    print(f"\n  Disparate Impact Ratio (Woman / Man): {disp_impact:.3f}")

    #  Equal Opportunity (difference in recall) 
    woman_acc = accuracy_table.loc["Woman"].mean() if "Woman" in accuracy_table.index else 0
    man_acc = accuracy_table.loc["Man"].mean() if "Man" in accuracy_table.index else 0
    equal_opp_diff = woman_acc - man_acc
    print(f"\n Equal Opportunity Difference (Woman - Man): {equal_opp_diff:.3f}")

    # Save detailed report
    with open(output_path, "w") as f:
        f.write("true_emotion\t" + "\t".join(accuracy_table.columns) + "\n")
        for gender in accuracy_table.index:
            row = "\t".join(f"{val:.3f}" for val in accuracy_table.loc[gender])
            f.write(f"{gender}\t{row}\n")
        f.write(f"\n  Disparate Impact Ratio (Woman / Man): {disp_impact:.3f}\n")
        f.write(f"\n Equal Opportunity Difference (Woman - Man): {equal_opp_diff:.3f}\n")

    print(f" Fairness report saved to '{output_path}'")

    # Plot gender × emotion accuracy
    plt.figure(figsize=(10, 6))
    accuracy_table.T.plot(kind="bar")
    plt.title("Emotion Classification Accuracy by Gender")
    plt.xlabel("Emotion")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.legend(title="Gender")
    plt.tight_layout()
    plt.show()

# Run audit
fairness_audit(pred_df)
