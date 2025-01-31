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
import random
np.random.seed(0)

# Import all classes except "BaseBenefitsProgram", "BenefitsProgramMeta"
module_name = "users.benefits_programs"
classes = import_all_classes(module_name)

benefits_classes = {}
min_line_nos = {}
max_line_nos = {}

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

    def get_demographic_df():
        """
        equivalent of `fuzz` but using the demographic randomizer and no filtering
        """


    def fuzz(limit, trials):
        assert trials >= limit

        source = inspect.getsource(users.benefits_programs)
        n_source_lines = len(source.splitlines()) + 1

        # Initialize the vector with zeros
        vector = [0 for _ in range(n_source_lines)]

        candidate_vectors = []
        hhs = []
        eligibilities = []
        program_liness = []
        for _ in tqdm(range(trials)):
            # Generate a new vector initialized with zeros
            new_vector = [0 for _ in range(n_source_lines)]

            # Get a random household input
            hh = get_random_household_input()
            hhs.append(hh)
            eligibility = {}
            program_lines = {}
            # Loop through all the classes to compute new_vector
            for class_name in benefits_classes.values():
                source_lines = list(
                    DatasetConstructor._trace_execution(
                        class_name.__call__, hh
                    ).values()
                )[0]
                # min_line_nos[class_name.__name__] = min(source_lines + [min_line_nos[class_name.__name__]])
                # max_line_nos[class_name.__name__] = max(source_lines + [max_line_nos[class_name.__name__]])

                for line in source_lines:
                    new_vector[int(line)] = 1
                program_lines[class_name.__name__] = source_lines

                eligibility[class_name.__name__] = class_name.__call__(hh)

            eligibilities.append(eligibility)
            candidate_vectors.append(new_vector)
            program_liness.append(program_lines)
            # program_lines

        # name_line_dict = {}
        # for name, min_line, max_line in zip(benefits_classes.keys(), min_line_nos.values(), max_line_nos.values()):
        #     name_line_dict[name] = list(range(min_line, max_line + 1))
        # line_name_dict = {}
        # for k, v in name_line_dict.items():
        #     for i in v:
        #         line_name_dict[i] = k

        elig_df = pd.DataFrame(eligibilities)
        num_passes = elig_df.sum(axis=0)
        non_passing_programs = elig_df.columns[num_passes == 0]
        print(f"Non-passing programs: {non_passing_programs}")
        t = torch.Tensor(candidate_vectors)
        # drop indices where all elements are 0 or 1
        t = t[:, t.sum(dim=0) != 0]
        t = t[:, t.sum(dim=0) != trials]

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
        print(f"found cover of size {len(hh_cover)}")

        all_program_names = list(benefits_classes.keys())
        shrinking = []
        for i in indices:
            shrinking.append(
                {
                    "hh": hhs[i],
                    "eligibility": eligibilities[i],
                    "program_lines": program_liness[i],
                    "programs": set(all_program_names),
                }
            )
        df = pd.DataFrame(shrinking)

        # iterate over the df, dropping programs from "programs" if they do not contribute unique lines in "program_lines"
        changed = True
        while changed:
            changed = False
            for p in all_program_names:
                other_hh_used_lines = set()
                # for i in range(len(df)):
                # if p in df.iloc[i]["programs"]:
                indices = list(range(len(df)))
                random.shuffle(indices)
                for i in indices:
                    if p in df.iloc[i]["programs"]:
                        # used_lines = used_lines | set(df.iloc[i]["program_lines"][p])
                        other_lines_df = pd.concat([df.iloc[:i], df.iloc[i + 1 :]])
                        other_lines_programs = other_lines_df[
                            other_lines_df["programs"].apply(lambda x: p in x)
                        ]
                        other_lines = (
                            other_lines_programs["program_lines"]
                            .apply(lambda x: x[p])
                            .tolist()
                        )
                        other_hh_used_lines = set(
                            [item for sublist in other_lines for item in sublist]
                        )
                        this_hh_used_lines = set(df.iloc[i]["program_lines"][p])
                        this_hh_unique_lines = this_hh_used_lines - other_hh_used_lines
                        if len(this_hh_unique_lines) == 0:
                            print(f"removing program {p} from hh {i}")
                            df.iloc[i]["programs"].remove(p)
                            df.iloc[i]["program_lines"].pop(p)
                            changed = True
                            print(df["programs"].apply(lambda x: len(x)))
                # flatten the list of lists
                print


        # double check coverage
        original_df = pd.DataFrame(shrinking)
        final_used_lines_set = lambda d: set().union(
            *[
                set(
                    sum(
                        [
                            v
                            for k, v in d.iloc[i]["program_lines"].items()
                            if k in d.iloc[i]["programs"]
                        ],
                        [],
                    )
                )
                for i in range(len(d))
            ]
        )
        assert final_used_lines_set(df) == final_used_lines_set(original_df)

        df = df[df["programs"].apply(lambda x: len(x) > 0)]
        df["programs"] = df["programs"].apply(lambda x: list(x))
        return df
        # return hh_cover
