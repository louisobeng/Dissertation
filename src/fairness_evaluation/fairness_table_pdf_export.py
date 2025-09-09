import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors as mcolors

#  LOAD CSV 
csv_path = "/Users/mct/Documents/GitHub/dissertation/All_Scratch_gender_fairness_multimodel_results.csv"
df = pd.read_csv(csv_path)

# SELECT COLUMNS TO INCLUDE
selected_columns = [
    "Dataset", "Model", "Emotion",
    "Emotion_Acc", "Man_Acc", "Woman_Acc", 
    "Acc_Gap", "Bias_Flag", "F1_Gender", 
    "Disp_Impact", "Equal_Opp_Diff"
]

#  CHECK FOR MISSING COLUMNS 
missing = [col for col in selected_columns if col not in df.columns]
if missing:
    print("Missing columns:", missing)
    exit()

#  Format Table 
table_df = df[selected_columns].copy()
table_df = table_df.round(3)

#  Setup Figure 
fig, ax = plt.subplots(figsize=(24, min(0.5 * len(table_df), 20)))
ax.axis('off')

# Create Table
table = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 0.9)

# Highlight Bias Rows 
bias_col_index = selected_columns.index("Bias_Flag")
for i, row in enumerate(table_df.values):
    if row[bias_col_index] == "Bias_Detected!":
        for j in range(len(selected_columns)):
            cell = table[i + 1, j]
            cell.set_facecolor(mcolors.to_rgba("lightcoral", 0.3))

#  Save to PDF 
output_pdf_path = "/Users/mct/Documents/GitHub/dissertation/All_main_gender_fairness_table_landscape.pdf"
with PdfPages(output_pdf_path) as pdf:
    pdf.savefig(fig, bbox_inches='tight')
print(f" Landscape table PDF saved to: {output_pdf_path}")
