import os
import sys
import json
import numpy as np


def process_directory(root_dir):
    counts = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "history.jsonl" in filenames:
            file_path = os.path.join(dirpath, "history.jsonl")
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            lines = content.splitlines()
            hist = [json.loads(line) for line in lines]
            for x in hist:
                count = 0
                dialog = x["dialog"]
                for turn in dialog:
                    role = turn[-1]["role"]
                    if role == "answer_cq":
                        count += 1
                counts.append(count)
            count_mean = np.mean(counts)
            count_std = np.std(counts)
            count_file_path = os.path.join(dirpath, "count.txt")
            with open(count_file_path, "w", encoding="utf-8") as count_file:
                json.dump({"mean": count_mean, "std": count_std}, count_file, indent=4)
            print(f"mean: {count_mean:.2f}, std: {count_std:.2f}")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    process_directory(root)
