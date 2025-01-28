from copy import deepcopy
from pathlib import PurePath
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support


def plot_code_mode_results(
    df: pd.DataFrame,
    labels_df: pd.DataFrame,
    output_dir,
    experiment_params: dict = {},
):
    """
    Plot results of code mode experiment with additional metrics (F1, precision, recall).
    Counts all NaN predictions as incorrect.
    """
    n = len(df)
    p_names = df.columns
    dfs = {}
    labels_df = labels_df.astype(int)
    output_dir = PurePath(output_dir)
    for p in p_names:
        for i in range(n):
            targets = ~df[p].isna()
            preds = df[targets][p]
            labels = labels_df[targets][p]
            correct = preds == labels
            tp = labels & correct
            tn = ~labels & correct
            fp = labels & ~correct
            fn = ~labels & ~correct
            dfs[p] = pd.DataFrame(
                {
                    "preds": preds,
                    "labels": labels,
                    "correct": correct,
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "program": [p] * len(preds),
                    "i": [i] * len(preds),
                }
            )
    # make one main df containing i, program, label, pred
    main_df = pd.concat(dfs.values())
    acc = main_df["correct"].mean()
    tps = main_df["tp"].sum()
    tns = main_df["tn"].sum()
    fps = main_df["fp"].sum()
    fns = main_df["fn"].sum()
    precision = tps / (tps + fps)
    recall = tps / (tps + fns)
    f1 = 2 * (precision * recall) / (precision + recall)
    summary_df = pd.DataFrame(
        {
            "accuracy": [acc],
            "precision": [precision],
            "recall": [recall],
            "f1": [f1],
            "tps": [tps],
            "tns": [tns],
            "fps": [fps],
            "fns": [fns],
        }
    )

    # dump results to csv
    summary_df.to_csv(output_dir / "results_sum.csv")
    main_df.to_csv(output_dir / "results_all.csv")


# from typing import List, Dict
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import numpy as np
# from datetime import datetime
# from pathlib import PurePath
# from copy import deepcopy


# def plot_metrics_per_turn(
#     predictions: list[list[bool]],
#     labels: pd.DataFrame,
#     last_turn_iteration,
#     output_dir,
#     experiment_params: Dict = {},
# ) -> None:
#     """
#     Plot metrics per turn
#     """
#     output_dir = PurePath(output_dir)
#     now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
#     last_turn_iteration = pd.Series(last_turn_iteration)
#     # replace `None` predictions with the last non-none prediction
#     # last_turn_iteration = np.array(last_turn_iteration)
#     # for i in range(len(predictions)):
#     #     for j in range(len(predictions[i])):
#     #         if predictions[i][j] is None:
#     #             predictions[i][j] = predictions[i][j - 1]
#     predictions_dfs = []
#     for i in range(len(predictions[0])):
#         predictions_df = pd.DataFrame(list(np.array(predictions).T[i]))
#         predictions_dfs.append(predictions_df)

#     metrics = {
#         "accuracy": [],
#         "precision": [],
#         "recall": [],
#         "f1": [],
#     }

#     # calculate metrics per turn
#     for i in range(len(predictions[0])):
#         metrics["accuracy"].append(
#             accuracy_score(labels.values.flatten(), predictions_dfs[i].values.flatten())
#         )
#         metrics["precision"].append(
#             precision_score(
#                 labels.values.flatten(), predictions_dfs[i].values.flatten()
#             )
#         )
#         metrics["recall"].append(
#             recall_score(labels.values.flatten(), predictions_dfs[i].values.flatten())
#         )
#         metrics["f1"].append(
#             f1_score(labels.values.flatten(), predictions_dfs[i].values.flatten())
#         )
#         print
#     # plot a bar graph of when the agents declared certainty
#     num_remaining = []  # num remaining after each turn
#     for i in range(len(predictions[0])):
#         num_remaining.append((last_turn_iteration >= i).mean())

#     sns.lineplot(data=metrics, palette=sns.color_palette("husl", 8))
#     sns.barplot(data=num_remaining, color="green", alpha=0.3)
#     plt.xlabel("Turn")
#     plt.ylabel("Accuracy")
#     plt.xticks(range(0, len(predictions[0]), 5))
#     # add additional space for the caption
#     plt.subplots_adjust(bottom=0.5)
#     if experiment_params:
#         caption_text = "\n".join([f"{k}: {v}" for k, v in experiment_params.items()])
#         plt.figtext(
#             0.02,
#             0.02,
#             caption_text,
#             wrap=True,
#             horizontalalignment="center",
#             fontsize=6,
#             ha="left",
#         )

#     plt.show()
#     # save the plot
#     model = experiment_params["Backbone Model"].replace("/", "_")
#     programs = "_".join(experiment_params["Programs"].split(", "))
#     # drop all lowercase letters in programs
#     programs = "".join([i for i in programs if not i.islower()])
#     plt.savefig(
#         output_dir / f"mpt_{model}_n={experiment_params['Downsample Size']}.png",
#         dpi=300,
#     )
#     pd.DataFrame(metrics).to_json(
#         output_dir / f"mpt_{model}_n={experiment_params['Downsample Size']}.jsonl",
#         lines=True,
#         orient="records",
#     )
#     pass


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
