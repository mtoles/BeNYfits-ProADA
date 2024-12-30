import trace
import inspect
from scipy.spatial.distance import cityblock
from collections import defaultdict
from users.benefits_programs import ChildAndDependentCareTaxCredit, EarlyHeadStartPrograms, InfantToddlerPrograms, ComprehensiveAfterSchool, InfantToddlerPrograms, ChildTaxCredit, DisabilityRentIncreaseExemption, EarnedIncomeTaxCredit, HeadStart, get_random_household_input
from tqdm import tqdm
from collections import defaultdict

class DatasetConstructor:
    def _trace_execution(func, *args, **kwargs):
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
            if count > 0 and filename.endswith(function_filename):  # Only include lines that were executed
                executed_lines[filename].append(line_number)

        return executed_lines
    
    def fuzz(limit=20, trials=10**3):
        
        # Number of lines in the source code of the function
        n_source_lines = 1000

        # Initialize the vector with zeros
        vector = [0 for _ in range(n_source_lines)]

        # Output list to store households with maximal distance
        output = []

        # Variable to track the number of iterations
        iteration_count = 0

        for _ in tqdm(range(limit)):
            max_distance = -1
            max_new_vector = None
            max_hh = None

            for _ in range(trials):
                # Generate a new vector initialized with zeros
                new_vector = [0 for _ in range(n_source_lines)]

                # Get a random household input
                hh = get_random_household_input()

                # Loop through all the classes to compute new_vector
                for class_name in [ChildAndDependentCareTaxCredit, EarlyHeadStartPrograms, InfantToddlerPrograms, ComprehensiveAfterSchool, InfantToddlerPrograms, ChildTaxCredit, DisabilityRentIncreaseExemption, EarnedIncomeTaxCredit, HeadStart]:
                    source_lines = list(DatasetConstructor._trace_execution(class_name.__call__, hh).values())[0]

                    for line in source_lines:
                        new_vector[int(line)] = 1

                # Compute the Manhattan distance (cityblock distance) between vector and new_vector
                distance = cityblock(vector, new_vector)

                # Update the maximum distance, vector, and household input if applicable
                if distance > max_distance:
                    max_distance = distance
                    max_new_vector = new_vector
                    max_hh = hh

            # Increment iteration count
            iteration_count += 1

            # Update the vector using an incremental average to maintain linear complexity
            vector = [(vector[i] * (iteration_count - 1) + max_new_vector[i]) / iteration_count for i in range(n_source_lines)]

            # Append the maximally distant household input to the output list
            output.append(max_hh)

        # Return the output list
        return output
