import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set font size for plots
plt.rcParams.update({'font.size': 16})

# List of violin plot result analysis files with their metadata
violin_files = [
    {
        "path": "results/violin/violin_heating_32.xlsx",
        "title": "Actual vs Predicted Heating Load (Batch Size: 32)",
        "ylabel": "Heating Load",
        "actual_col": "Actual Heating Load"
    },
    {
        "path": "results/violin/violin_heating_64.xlsx",
        "title": "Actual vs Predicted Heating Load (Batch Size: 64)",
        "ylabel": "Heating Load",
        "actual_col": "Actual Heating Load"
    },
    {
        "path": "results/violin/violin_cooling_32.xlsx",
        "title": "Actual vs Predicted Cooling Load (Batch Size: 32)",
        "ylabel": "Cooling Load",
        "actual_col": "Actual Cooling Load"
    },
    {
        "path": "results/violin/violin_cooling_64.xlsx",
        "title": "Actual vs Predicted Cooling Load (Batch Size: 64)",
        "ylabel": "Cooling Load",
        "actual_col": "Actual Cooling Load"
    }
]

# Loop through each file and generate violin plots
for vf in violin_files:
    if os.path.exists(vf["path"]):
        df = pd.read_excel(vf["path"])

        # Convert Arabic decimal separator if present
        df = df.applymap(lambda x: str(x).replace('٫', '.')).astype(float)

        # Select relevant columns
        relevant_data = df[[vf["actual_col"], 'DNN', 'CNN', 'LSTM', 'CNN-LSTM']]

        # Plot the violin chart
        plt.figure(figsize=(12, 8))
        sns.violinplot(data=relevant_data)
        plt.title(vf["title"])
        plt.ylabel(vf["ylabel"])
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print(f"File not found: {vf['path']}")