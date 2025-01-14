import sys
import trace
import inspect
from scipy.spatial.distance import cityblock
from collections import defaultdict
from users.benefits_programs import (
    get_random_household_input,
)
import users.benefits_programs
from tqdm import tqdm

from utils import import_all_classes
from sklearn_extra.cluster import KMedoids
import torch
import pandas as pd
import numpy as np

np.random.seed(0)

# Import all classes except "BaseBenefitsProgram", "BenefitsProgramMeta"
module_name = "users.benefits_programs"
classes = import_all_classes(module_name)

benefits_classes = {}

for class_name, cls in classes.items():
    if class_name not in ["BaseBenefitsProgram", "BenefitsProgramMeta"]:
        benefits_classes[class_name] = cls


class DatasetConstructor:
    def _trace_execution(func, *args, **kwargs):
        original_tracer = sys.gettrace()
        try:
            # Get the filename where the function is defined
            function_filename = inspect.getfile(func)

            # Create a Trace object for tracking execution
            tracer = trace.Trace(count=True, trace=False)

            # Run the function with tracing
            tracer.runfunc(func, *args, **kwargs)

            # Retrieve results from the tracer
            results = tracer.results()

            # Extract executed lines by file and line number
            executed_lines = defaultdict(list)

            for (filename, line_number), count in results.counts.items():
                if count > 0 and filename.endswith(
                    function_filename
                ):  # Only include lines that were executed
                    executed_lines[filename].append(line_number)

            return executed_lines
        finally:
            sys.settrace(original_tracer)

    def fuzz(limit, trials):
        assert trials >= limit

        source = inspect.getsource(users.benefits_programs)
        n_source_lines = len(source.splitlines()) + 1

        # Initialize the vector with zeros
        vector = [0 for _ in range(n_source_lines)]

        candidate_vectors = []
        hhs = []
        eligibilities = []
        for _ in tqdm(range(trials)):
            # Generate a new vector initialized with zeros
            new_vector = [0 for _ in range(n_source_lines)]

            # Get a random household input
            hh = get_random_household_input()
            hhs.append(hh)
            eligibility = {}
            # Loop through all the classes to compute new_vector
            for class_name in benefits_classes.values():
                source_lines = list(
                    DatasetConstructor._trace_execution(
                        class_name.__call__, hh
                    ).values()
                )[0]

                for line in source_lines:
                    new_vector[int(line)] = 1

                eligibility[class_name.__name__] = class_name.__call__(hh)

            eligibilities.append(eligibility)
            candidate_vectors.append(new_vector)

        elig_df = pd.DataFrame(eligibilities)
        num_passes = elig_df.sum(axis=0)
        non_passing_programs = elig_df.columns[num_passes == 0]
        print(f"Non-passing programs: {non_passing_programs}")
        t = torch.Tensor(candidate_vectors)
        # drop indices where all elements are 0 or 1
        t = t[:, t.sum(dim=0) != 0]
        t = t[:, t.sum(dim=0) != trials]
        # # run medoids
        # kmedoids = KMedoids(n_clusters=limit, random_state=0)
        # kmedoids.fit(t)
        # medoids = torch.Tensor(kmedoids.cluster_centers_)
        best_starting_i = t.sum(dim=0).max().int()
        cover = t[best_starting_i].unsqueeze(0)
        indices = [best_starting_i]

        def add_best_new_vector(cover, candidates, indices):
            empty_indices = cover.sum(dim=0) == 0
            if empty_indices.any():
                candidates_nz = candidates[:, empty_indices]
                best_candidate_i = candidates_nz.sum(dim=1).argmax()
                indices.append(best_candidate_i)
                best_candidate = candidates[best_candidate_i]
                new_cover = torch.concat([cover, best_candidate.unsqueeze(0)], dim=0)
                percent_coverage = new_cover.any(dim=0).float().mean()
                print(f"Coverage: {percent_coverage:.4f}")
                return new_cover, indices
            else:
                print("Done")
                return cover, indices

        for i in range(limit):
            cover, indices = add_best_new_vector(cover, t, indices)

        hh_cover = [hhs[i] for i in indices]
        return hh_cover
        # greedily add vectors to cover

        # percent_lines_hit_by_medoids = (medoids.mean(axis=0)==0).float().mean()
        # print

        #         # Compute the Manhattan distance (cityblock distance) between vector and new_vector
        #         distance = cityblock(vector, new_vector)

        #         # Update the maximum distance, vector, and household input if applicable
        #         if distance > max_distance:
        #             max_distance = distance
        #             max_new_vector = new_vector
        #             max_hh = hh

        #     # Increment iteration count
        #     iteration_count += 1

        #     # Update the vector using an incremental average to maintain linear complexity
        #     vector = [
        #         (vector[i] * (iteration_count - 1) + max_new_vector[i])
        #         / iteration_count
        #         for i in range(n_source_lines)
        #     ]

        #     # Append the maximally distant household input to the output list
        #     output.append(max_hh)

        # # Return the output list
        # return output
