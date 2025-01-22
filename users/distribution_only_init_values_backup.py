from users import Person
import matplotlib.pyplot as plt
import numpy as np
from user_features import (
    PersonAttributeMeta
)

def sample_and_plot_distributions(num_samples=1000, output_dir="./users/output"):
    """
    1. Retrieves all attributes with a distribution.
    2. Samples each attribute `num_samples` times using `uniform()`.
    3. Plots a bar chart of the resulting sample proportions.
    4. Annotates each bar with the original (initial) probability from the distribution.
    
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
        
        print(f"Attribute: {attr}")
        print(f"Samples: {samples}")
        
        # -----------------------------------------
        # 2. Build a label->prob mapping (from .distribution)
        #    For continuous/interval attributes (like 'age'), 
        #    we label each interval as "start-end".
        # -----------------------------------------
        def get_label_and_prob(d):
            # d is either (("start","end"), probability) or (value, probability)
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
            label_prob[label] = prob

        # -----------------------------------------
        # 3. Tally up the sampled results
        #    For intervals, find which bin each sample falls into.
        # -----------------------------------------
        freq_dict = {lbl: 0 for lbl in label_prob.keys()}
        
        def sample_to_label(s):
            # If this attribute is an interval-based distribution (e.g. 'age'),
            # we figure out which interval s falls into.
            # Otherwise, if it's discrete (like 'sex'), s is already the category.
            if isinstance(dist[0][0], tuple):
                # We have intervals of form ((start, end), prob)
                for (rng, _prob) in dist:
                    start, end = rng
                    # Make sure s is in [start, end]
                    if start <= s <= end:
                        return f"{start}-{end}"
            else:
                # Discrete distribution: just convert to string
                return str(s)

        # Count how many samples in each bin/category
        for s in samples:
            freq_dict[sample_to_label(s)] += 1

        # -----------------------------------------
        # 4. Prepare data for plotting
        # -----------------------------------------
        labels = list(freq_dict.keys())         # category labels (e.g. '0-4', 'male', etc.)
        freqs = [freq_dict[lbl] for lbl in labels]
        total = sum(freqs)
        proportions = [f / total for f in freqs]

        # -----------------------------------------
        # 5. Plot the bar chart
        # -----------------------------------------
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Draw bars
        bars = ax.bar(x, proportions, color="skyblue", edgecolor="black")

        # Basic labeling
        ax.set_title(f"Distribution of '{attr}' from {num_samples} samples")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel("Proportion")
        ax.set_ylim([0, 1.0])  # proportions go from 0 to 1

        # -----------------------------------------
        # 6. Annotate each bar with the original distribution value
        #    We'll show it right above the bar.
        # -----------------------------------------
        for i, lbl in enumerate(labels):
            # initial distribution for this label
            init_prob = label_prob[lbl]  
            # current bar's height
            bar_height = proportions[i]
            
            # Put text slightly above the top of the bar
            ax.text(
                x[i], 
                bar_height + 0.01,  # shift text a bit above the bar
                f"Init: {init_prob}%", 
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

sample_and_plot_distributions(num_samples=10)