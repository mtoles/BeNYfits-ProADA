from typing import List, Dict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from datetime import datetime
from pathlib import PurePath


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
    plt.subplots_adjust(bottom=0.2)
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
    model = experiment_params["Backbone Model"]
    programs = "_".join(experiment_params["Programs"].split(", "))
    # drop all lowercase letters in programs
    programs = "".join([i for i in programs if not i.islower()])
    plt.savefig(output_dir / f"mpt_{model}_{programs}_n={experiment_params['Downsample Size']}.png", dpi=300)
    pass
