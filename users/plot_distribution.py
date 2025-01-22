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
    
    :param num_samples: Number of samples to draw for each attribute
    :param output_dir: Directory to save the resulting bar charts
    """
    # Get only those attributes that actually have a defined distribution
    dist_attrs = PersonAttributeMeta.attribute_distribution()

    for attr, dist in dist_attrs.items():
        # -----------------------------------------
        # 1. Collect samples
        # -----------------------------------------
        samples = [PersonAttributeMeta.registry[attr].uniform() for _ in range(num_samples)]
        
        # -----------------------------------------
        # 2. Build a label->prob mapping (from .distribution)
        #    For continuous/interval attributes (like 'age'),
        #    we label each interval as "start-end".
        # -----------------------------------------
        def get_label_and_prob(d):
            # d is either ((start,end), probability) or (value, probability)
            value, prob = d
            if isinstance(value, tuple):
                start, end = value
                label = f"{start}-{end}"
            else:
                label = str(value)
            return label, prob

        label_prob = {}
        for d in dist:
            label, prob = get_label_and_prob(d)
            label_prob[label] = prob  # prob is in %, e.g. 5.4 means 5.4%

        # -----------------------------------------
        # 3. Tally up the sampled results
        #    For intervals, find which bin each sample falls into.
        # -----------------------------------------
        freq_dict = {lbl: 0 for lbl in label_prob.keys()}
        
        def sample_to_label(s):
            """
            For interval-based distribution (e.g. 'age'), find which interval
            s belongs to. Otherwise, return the discrete label (like 'male').
            """
            if isinstance(dist[0][0], tuple):
                # intervals of form ((start, end), prob)
                for (rng, _prob) in dist:
                    start, end = rng
                    if start <= s <= end:
                        return f"{start}-{end}"
            else:
                # Discrete distribution
                return str(s)

        for s in samples:
            freq_dict[sample_to_label(s)] += 1

        # -----------------------------------------
        # 4. Prepare data for plotting
        # -----------------------------------------
        labels = list(freq_dict.keys())          # category labels (e.g. '0-4', 'male', etc.)
        freqs = [freq_dict[lbl] for lbl in labels]
        total = sum(freqs)
        proportions = [f / total for f in freqs]  # fraction in [0,1]

        # -----------------------------------------
        # 5. Plot the bar chart
        # -----------------------------------------
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Draw bars
        bars = ax.bar(x, proportions, color="skyblue", edgecolor="black")

        # Title & labels
        ax.set_title(f"Distribution of '{attr}' from {num_samples} samples")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel("Proportion")
        ax.set_ylim([0, 1.0])  # proportions go from 0 to 1

        # -----------------------------------------
        # 6. Annotate each bar:
        #    - Sample proportion (based on the simulation)
        #    - Initial distribution probability (from label_prob)
        # -----------------------------------------
        for i, lbl in enumerate(labels):
            init_prob = label_prob[lbl]          # e.g. 5.4 means 5.4%
            sample_prop = proportions[i]         # fraction in [0,1]
            sample_percent = sample_prop * 100   # convert fraction to percentage

            # Position the text slightly above the bar
            bar_height = sample_prop
            text_x = x[i]
            text_y = bar_height + 0.01

            annotation = (f"Sample: {sample_percent:.1f}%\n"
                          f"Init: {init_prob:.1f}%")
            
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

        # -----------------------------------------
        # 7. Save the figure
        # -----------------------------------------
        plot_filename = f"{output_dir}/{attr}_distribution.png"
        plt.savefig(plot_filename, dpi=150)
        print(f"Saved distribution plot for '{attr}' to: {plot_filename}")
        plt.close(fig)

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    sample_and_plot_distributions(num_samples=1000)