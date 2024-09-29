from typing import List, Dict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def plot_metrics_per_turn(
    predictions: list[list[bool]], labels: pd.DataFrame, last_turn_iteration
) -> None:
    """
    Plot metrics per turn
    """
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
    }

    # calculate metrics per turn
    for i in range(len(predictions[0])):
        metrics["accuracy"].append(
            accuracy_score(labels.values.flatten(), predictions_dfs[i].values.flatten())
        )
    # plot a bar graph of when the agents declared certainty
    num_remaining = []  # num remaining after each turn
    for i in range(len(predictions[0])):
        num_remaining.append((last_turn_iteration >= i).mean())

    sns.lineplot(data=metrics, palette=sns.color_palette("husl", 8))
    sns.barplot(data=num_remaining, color="green", alpha=0.3)
    plt.xlabel("Turn")
    plt.ylabel("Accuracy")
    plt.show()
    # save the plot
    plt.savefig("metrics_per_turn.png")
    pass
