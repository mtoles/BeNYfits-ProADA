from users import Person
import matplotlib.pyplot as plt
import numpy as np
from user_features import (
    PersonAttributeMeta
)
import random

def sample_and_plot_distributions(num_samples=10, output_dir="./users/output"):
    """
    1. Retrieves all attributes with a distribution.
    2. Samples each attribute `num_samples` times using `uniform()`.
    3. Plots a bar chart of the resulting sample proportions.
    4. Annotates each bar with both:
       - The original (initial) distribution probability
       - The sampled proportion.
       
    Special handling:
       - If `.uniform()` returns booleans (True/False) but distribution is
         labeled ("Yes", p1), ("No", p2), we map True -> "Yes", False -> "No".
    
    :param num_samples: Number of samples to draw for each attribute
    :param output_dir:  Directory to save the resulting bar charts
    """
    dist_attrs = PersonAttributeMeta.attribute_distribution()

    for attr, dist in dist_attrs.items():
        # ---------------------------------------------------
        # 1. Collect N samples from .uniform()
        # ---------------------------------------------------
        samples = [PersonAttributeMeta.registry[attr].uniform() for _ in range(num_samples)]
        
        # ---------------------------------------------------
        # 2. Extract label->prob (from distribution)
        #
        #    For discrete attributes, you might have:
        #       [("male", 47.5), ("female", 52.5)]
        #
        #    For intervals, you might have:
        #       [((0,4), 5.4), ((5,9), 5.4), ...]
        #
        #    For yes/no booleans:
        #       [("Yes", 0.74), ("No", 99.26)]
        # ---------------------------------------------------
        def get_label_and_prob(d):
            # d is either ((start,end), prob) or (str, prob)
            value, prob = d
            if isinstance(value, tuple):
                start, end = value
                label = f"{start}-{end}"
            else:
                label = str(value)
            return label, prob

        label_prob = {}
        for d in dist:
            label, prob = get_label_and_prob(d)   # e.g. ("Yes", 0.74)
            label_prob[label] = prob             # store probability (0.74, etc.)

        # ---------------------------------------------------
        # 3. Tally up the sampled results
        #    - If we have intervals, find which interval the sample falls into.
        #    - If sample is boolean, map True->"Yes", False->"No" if that matches
        #      the distribution keys.
        #    - Otherwise, do str(sample).
        # ---------------------------------------------------
        freq_dict = {lbl: 0 for lbl in label_prob.keys()}
        
        def sample_to_label(s):
            # Check if the distribution is interval-based
            if isinstance(dist[0][0], tuple):
                # intervals of form ((start, end), prob)
                for (rng, _prob) in dist:
                    start, end = rng
                    if start <= s <= end:
                        return f"{start}-{end}"
            
            # If sample is boolean and the distribution is "Yes"/"No"
            # we match accordingly:
            if isinstance(s, bool) and "Yes" in label_prob and "No" in label_prob:
                return "Yes" if s else "No"
            
            # Otherwise treat as a discrete category (string)
            return str(s)

        for s in samples:
            freq_dict[sample_to_label(s)] += 1

        # ---------------------------------------------------
        # 4. Prepare data for plotting
        # ---------------------------------------------------
        labels = list(freq_dict.keys())          # e.g. ["Yes","No"] or intervals, etc.
        freqs = [freq_dict[lbl] for lbl in labels]
        total = sum(freqs)
        proportions = [f / total for f in freqs]  # fraction in [0,1]

        # ---------------------------------------------------
        # 5. Plot the bar chart
        # ---------------------------------------------------
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8, 5))
        
        bars = ax.bar(x, proportions, color="skyblue", edgecolor="black")

        ax.set_title(f"Distribution of '{attr}' from {num_samples} samples")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel("Proportion")
        ax.set_ylim([0, 1.0])  # proportions range from 0 to 1

        # ---------------------------------------------------
        # 6. Annotate each bar with:
        #    - The sample proportion
        #    - The initial distribution probability
        # ---------------------------------------------------
        for i, lbl in enumerate(labels):
            init_prob = label_prob[lbl]          # e.g. 0.74
            sample_prop = proportions[i]         # fraction in [0,1]
            sample_percent = sample_prop * 100   # e.g. 55.2 for 0.552

            # text position
            bar_height = sample_prop
            text_x = x[i]
            text_y = bar_height + 0.01

            annotation = (f"Sample: {sample_percent:.2f}%\n"  # show 2 decimal places
                          f"Init: {init_prob:.2f}%")
            
            ax.text(
                text_x,
                text_y,
                annotation,
                ha='center',
                va='bottom',
                rotation=90,
                fontsize=8
            )

        plt.tight_layout()

        # ---------------------------------------------------
        # 7. Save the figure
        # ---------------------------------------------------
        plot_filename = f"{output_dir}/{attr}_distribution.png"
        plt.savefig(plot_filename, dpi=150)
        print(f"Saved distribution plot for '{attr}' to: {plot_filename}")
        plt.close(fig)

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    sample_and_plot_distributions(num_samples=10)