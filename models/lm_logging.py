from pathlib import PurePath
from typing import List
import json
import os
from copy import deepcopy
import jsonlines
import pandas as pd
from users import Household
from dataset_procedural import show_household

class LmLogger:
    """
    self.log is a list of conversations:
    [
        {
            "labels": {"ChildTaxCredit": True, "ComprehensiveAfterSchool": False},
            "dialog": [
                [dialog_history],
                [dialog_history],
                ...
            ],
            "predictions": [
                {"ChildTaxCredit": True, "ComprehensiveAfterSchool": False},
                {"ChildTaxCredit": True, "ComprehensiveAfterSchool": False} // one per dialog
            ],
        }
    ]
    """

    def __init__(self, log_dir):
        self.log = []  # list of conversations
        self.log_dir = log_dir
        self.history_path = PurePath(self.log_dir) / "history.jsonl"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.history_path):
            with open(self.history_path, "w") as f:
                f.write("")

    def add_empty_convo(self, labels):
        self.log.append({"labels": labels, "dialog": [], "predictions": []})

    def log_io(self, lm_input: List[dict], lm_output: str, role: str):
        # append to the history file
        # lm_input = deepcopy(lm_input)
        convo = lm_input + [{"role": role, "content": lm_output}]
        self.log[-1]["dialog"].append(convo)
        pass

    def log_predictions(self, predictions: List[dict]):
        self.log[-1]["predictions"].extend(predictions)

    def log_hh_diff(self, hh: Household):
        self.log[-1]["hh_diff"] = show_household(hh)

    def save(self):
        # self.log[0]["dialog"] = self.log[0]["dialog"]
        with open(self.history_path, "w") as f:
            for convo in self.log:
                line = json.dumps(convo) + "\n"
                f.write(line)
        pass
        #
