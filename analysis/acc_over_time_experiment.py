from typing import List, Dict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from datetime import datetime
from pathlib import PurePath
from copy import deepcopy

def plot_metrics_per_turn(
    predictions: list[list[bool]],
    labels: pd.DataFrame,
    last_turn_iteration,
    output_dir,
    experiment_params: Dict = {},
) -> None:
    """
    Plot metrics per turn
    """
    output_dir = PurePath(output_dir)
    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    last_turn_iteration = pd.Series(last_turn_iteration)
    # replace `None` predictions with the last non-none prediction
    # last_turn_iteration = np.array(last_turn_iteration)
    # for i in range(len(predictions)):
    #     for j in range(len(predictions[i])):
    #         if predictions[i][j] is None:
    #             predictions[i][j] = predictions[i][j - 1]
    predictions_dfs = []
    for i in range(len(predictions[0])):
        predictions_df = pd.DataFrame(list(np.array(predictions).T[i]))
        predictions_dfs.append(predictions_df)

    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    # calculate metrics per turn
    for i in range(len(predictions[0])):
        metrics["accuracy"].append(
            accuracy_score(labels.values.flatten(), predictions_dfs[i].values.flatten())
        )
        metrics["precision"].append(
            precision_score(
                labels.values.flatten(), predictions_dfs[i].values.flatten()
            )
        )
        metrics["recall"].append(
            recall_score(labels.values.flatten(), predictions_dfs[i].values.flatten())
        )
        metrics["f1"].append(
            f1_score(labels.values.flatten(), predictions_dfs[i].values.flatten())
        )
        print
    # plot a bar graph of when the agents declared certainty
    num_remaining = []  # num remaining after each turn
    for i in range(len(predictions[0])):
        num_remaining.append((last_turn_iteration >= i).mean())

    sns.lineplot(data=metrics, palette=sns.color_palette("husl", 8))
    sns.barplot(data=num_remaining, color="green", alpha=0.3)
    plt.xlabel("Turn")
    plt.ylabel("Accuracy")
    plt.xticks(range(0, len(predictions[0]), 5))
    # add additional space for the caption
    plt.subplots_adjust(bottom=0.5)
    if experiment_params:
        caption_text = "\n".join([f"{k}: {v}" for k, v in experiment_params.items()])
        plt.figtext(
            0.02,
            0.02,
            caption_text,
            wrap=True,
            horizontalalignment="center",
            fontsize=6,
            ha="left",
        )

    plt.show()
    # save the plot
    model = experiment_params["Backbone Model"].replace("/", "_")
    programs = "_".join(experiment_params["Programs"].split(", "))
    # drop all lowercase letters in programs
    programs = "".join([i for i in programs if not i.islower()])
    plt.savefig(
        output_dir / f"mpt_{model}_n={experiment_params['Downsample Size']}.png",
        dpi=300,
    )
    pd.DataFrame(metrics).to_json(
        output_dir / f"mpt_{model}_n={experiment_params['Downsample Size']}.jsonl",
        lines=True,
        orient="records",
    )
    pass


# def plot_code_mode_results(
#     df: pd.DataFrame,
#     labels: pd.DataFrame,
#     output_dir,
#     experiment_params: Dict = {},
# ):
#     """
#     Plot results of code mode experiment
#     """
#     output_dir = PurePath(output_dir)
#     p_names = df.columns
#     results = []
#     n = len(df)
#     for p in p_names:
#         # calculate % correct, % incorrect, % unknown (prediction is None)
#         p_NaN = df[p].isna().sum() / n
#         p_correct = (df[p].fillna(-1) == labels[p]).sum() / n
#         p_incorrect = (df[p].fillna(-1) != labels[p]).sum() / n - p_NaN

#         results.append(
#             pd.DataFrame(
#                 {
#                     "program": [p],
#                     "correct": [p_correct],
#                     "incorrect": [p_incorrect],
#                     "unknown": [p_NaN],
#                 }
#             )
#         )
#     results_df = pd.concat(results).set_index("program")
#     plt.bar(p_names, results_df["correct"], label="correct")
#     plt.bar(
#         p_names,
#         results_df["incorrect"],
#         bottom=results_df["correct"],
#         label="incorrect",
#     )
#     plt.bar(
#         p_names,
#         results_df["unknown"],
#         bottom=results_df["correct"] + results_df["incorrect"],
#         label="unknown",
#     )
#     plt.title("Code Mode Results")
#     plt.xlabel("Program")
#     plt.ylabel("Accuracy")
#     plt.xticks(range(len(p_names)), p_names, rotation=60, fontsize=6)
    
#     # Add additional space under the figure
#     plt.subplots_adjust(bottom=0.3)
    
#     displayed_params = deepcopy(experiment_params)
#     del displayed_params["Programs"]
#     if experiment_params:
#         caption_text = "\n".join([f"{k}: {v}" for k, v in displayed_params.items()])
#         plt.figtext(
#             0.02,
#             0.02,
#             caption_text,
#             wrap=True,
#             horizontalalignment="center",
#             fontsize=5,
#             ha="left",
#         )
#     plt.show()
#     # save the plot
#     model = experiment_params["Backbone Model"].replace("/", "_")
#     programs = "_".join(experiment_params["Programs"].split(", "))
#     # drop all lowercase letters in programs
#     programs = "".join([i for i in programs if not i.islower()])
#     plt.savefig(
#         output_dir / f"code_mode_{model}_n={experiment_params['Downsample Size']}.png",
#         dpi=300,
#     )
#     results_df.to_json(
#         output_dir
#         / f"code_mode_{model}_n={experiment_params['Downsample Size']}.jsonl",
#         lines=True,
#         orient="records",
#     )
#     print


from copy import deepcopy
from pathlib import PurePath
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

def plot_code_mode_results(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    output_dir,
    experiment_params: Dict = {},
):
    """
    Plot results of code mode experiment with additional metrics (F1, precision, recall).
    Counts all NaN predictions as incorrect.
    """
    output_dir = PurePath(output_dir)
    p_names = df.columns
    n = len(df)
    results = []

    for p in p_names:
        y_true = labels[p]
        y_pred = df[p].fillna(-1)
        
        # Compute correctness metrics
        p_correct = (y_pred == y_true).sum() / n
        p_incorrect = (y_pred != y_true).sum() / n

        tp = (y_pred == 1) & (y_true == 1)
        fp = (y_pred == 1) & (y_true == 0)
        fn = (y_pred == 0) & (y_true == 1)
        tn = (y_pred == 0) & (y_true == 0)

        # Compute precision, recall, F1 (macro averaged)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

        results.append(
            pd.DataFrame(
                {
                    "program_name": [p],
                    "correct": [p_correct],
                    "incorrect": [p_incorrect],
                    "f1": [f1],
                    "precision": [precision],
                    "recall": [recall],
                }
            )
        )
    results_df = pd.concat(results).set_index("program_name")

    fig, axs = plt.subplots(4, 1, figsize=(8, 12))
    # Accuracy plot
    axs[0].bar(p_names, results_df["correct"], label="correct")
    axs[0].bar(p_names, results_df["incorrect"], bottom=results_df["correct"], label="incorrect")
    axs[0].set_title("Accuracy Results")
    axs[0].set_xlabel("Program")
    axs[0].set_ylabel("Fraction")
    axs[0].set_xticks(range(len(p_names)))
    axs[0].set_xticklabels(p_names, rotation=60, fontsize=6)
    axs[0].legend()
    axs[0].set_ylim(0, 1)

    # Precision
    axs[1].bar(p_names, results_df["precision"])
    axs[1].set_title("Precision")
    axs[1].set_ylabel("Score")
    axs[1].set_xticks(range(len(p_names)))
    axs[1].set_xticklabels(p_names, rotation=60, fontsize=6)
    axs[1].set_ylim(0, 1)

    # Recall
    axs[2].bar(p_names, results_df["recall"])
    axs[2].set_title("Recall")
    axs[2].set_ylabel("Score")
    axs[2].set_xticks(range(len(p_names)))
    axs[2].set_xticklabels(p_names, rotation=60, fontsize=6)
    axs[2].set_ylim(0, 1)

    # F1
    axs[3].bar(p_names, results_df["f1"])
    axs[3].set_title("F1 Score")
    axs[3].set_ylabel("Score")
    axs[3].set_xticks(range(len(p_names)))
    axs[3].set_xticklabels(p_names, rotation=60, fontsize=6)
    axs[3].set_ylim(0, 1)

    plt.subplots_adjust(bottom=0.3, hspace=0.6)

    displayed_params = deepcopy(experiment_params)
    if "Programs" in displayed_params:
        del displayed_params["Programs"]

    if experiment_params:
        caption_text = "\n".join([f"{k}: {v}" for k, v in displayed_params.items()])
        plt.figtext(
            0.02,
            0.02,
            caption_text,
            wrap=True,
            horizontalalignment="left",
            fontsize=5,
        )

    plt.show()

    model = experiment_params["Backbone Model"].replace("/", "_")
    programs = "_".join(experiment_params["Programs"].split(", "))
    programs = "".join([i for i in programs if not i.islower()])

    fig.savefig(
        output_dir / f"code_mode_{model}_n={experiment_params['Downsample Size']}.png",
        dpi=300,
    )
    results_df.to_json(
        output_dir
        / f"code_mode_{model}_n={experiment_params['Downsample Size']}.jsonl",
        lines=True,
        orient="records",
    )

