import trace
import inspect
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
    
    def fuzz(func):
        pass