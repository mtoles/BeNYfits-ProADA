import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("paper_figs/fig_data.csv")

# --- Programmatically determine unique identifiers ---

# Unique datasets, strategies, and models from the CSV file
# unique_datasets = df["Dataset"].unique()  # e.g., ["Diversity", "Demographic"]
unique_strategies = df[
    "Strategy"
].unique()  # e.g., ["Direct Prompting", "ReAct", "ProADA (Ours)"]
unique_models = df[
    "Model"
].unique()  # e.g., ["Llama 3.1 8B", "Llama 3.1 70B", "GPT-4o", "GPT-4o mini"]
print(unique_models)
# Assign markers to models using a pre-defined list
marker_list = ["o", "s", "D", "X", "p", "2"]
model_markers = {
    model: marker_list[i % len(marker_list)] for i, model in enumerate(unique_models)
}

# Assign colors to strategies using a colormap
colors = ["#0f27b4", "#ff7f0e", "#2ca02c", "#d62728", "#7d1ea6", "#000000"]
strategy_colors = {
    strategy: colors[i % len(colors)] for i, strategy in enumerate(unique_strategies)
}

# colors = ["#000000", "#56B4E9", "#009E73", "#F0E442"]
# colors = [c_[2], c_[1], c_[0], c_[3]]
model_colors = {model: colors[i % len(colors)] for i, model in enumerate(unique_models)}


# If the F1 column contains percentage strings (like "78.7%"), strip the '%' sign and convert to float.
def parse_f1(val):
    if isinstance(val, str) and val.endswith("%"):
        return float(val.strip("%"))
    return float(val) * 100


df["F1"] = df["F1"].apply(parse_f1)
df["Turns"] = df["Turns"].astype(float)

# --- Create the scatter plot ---

fig, ax = plt.subplots(figsize=(5, 4.2))

# Set x-axis to logarithmic scale
# ax.set_xscale("log")

# Set specific x-axis ticks
ticks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ax.set_xticks(ticks)
ax.set_xticklabels([str(tick) for tick in ticks])

# To prevent duplicate legend entries, we track which labels have been added
legend_labels = set()

# Group the data by Dataset, Model, and Strategy so that each combination is plotted with its style
for (model, strategy), group in df.groupby(["Model", "Strategy"]):
    marker = model_markers[model]
    color = strategy_colors[strategy]
    label = f"{strategy} - {model}"

    # Only label the first time this combination appears
    if label in legend_labels:
        label_to_use = None
    else:
        label_to_use = label
        legend_labels.add(label)

    # Hollow markers (for Demographic)
    ax.scatter(
        group["Turns"],
        group["F1"],
        marker=marker,
        facecolors="none",
        edgecolors=color,
        label=label_to_use,
        linewidth=2,
        s=100,
        alpha=0.7,
    )

# Set bold axis labels
ax.set_xlabel("Dialog Turns", fontsize=12, fontweight="bold")
ax.set_ylabel("Average F1", fontsize=12, fontweight="bold")

# Add horizontal line for random baseline
ax.axhline(y=36.3, color="gray", linestyle=":", linewidth=1, label="Random")

# Adjust legend to be placed inside the plot area
handles, labels = ax.get_legend_handles_labels()

ax.legend(
    handles,
    labels,
    bbox_to_anchor=(0.98, 0.98),
    loc="upper right",
    fontsize="small",
)

plt.tight_layout(pad=0)

# Save the figure as a PDF file
pdf_filename = "paper_figs/main.png"
plt.savefig(pdf_filename, format="png", bbox_inches="tight")

# Display the plot
plt.show()

# Return the file path (if needed in your environment)
print("Saved plot to:", pdf_filename)
