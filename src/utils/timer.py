import time
import logging
import pandas as pd
from pathlib import Path
from contextlib import contextmanager

class MetricsTracker:
    """A utility class to track execution timing, log progress, and save metrics to CSV."""
    
    def __init__(self, logger: logging.Logger, pipeline_name: str = "Unknown", model_name: str = "Unknown"):
        self.logger = logger
        self.timing_data = {
            "Pipeline": pipeline_name,
            "model": model_name
        }
        
    @contextmanager
    def log_and_time(self, name: str):
        """Context manager to log and time a block of code and record it in timing_data."""
        self.logger.info(f"Starting: {name}...")
        start = time.time()
        yield
        duration = time.time() - start
        
        self.timing_data[name] = round(duration, 4)

    def get_duration(self, phase_name: str) -> float:
        """Helper to fetch the duration of a specific timed phase."""
        return self.timing_data.get(phase_name)

    def save_to_csv(self, data: dict, output_path: Path):
        """Saves dictionary data to a CSV file. Appends to existing file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format: if all values are lists, it's already rectangular data
        if all(isinstance(v, list) for v in data.values()):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
            
        file_exists = output_path.exists()
        write_header = not file_exists
        
        if file_exists:
            try:
                existing_cols = pd.read_csv(output_path, nrows=0).columns.tolist()
                if list(df.columns) != existing_cols:
                    # Schema mismatch (e.g. putting timing data in the same file as metrics)
                    write_header = True
            except Exception:
                write_header = True

        df.to_csv(output_path, mode='a', index=False, header=write_header)
            
        self.logger.info(f"Saved results to {output_path}")

    def print_timing_summary(self):
        """Logs a summary of all recorded timing phases."""
        self.logger.info("--- TIMING SUMMARY ---")
        for p, d in self.timing_data.items():
            self.logger.info(f"{p:<30}: {d} seconds")
            
    def print_metrics(self, title: str, metrics: dict):
        """Prints a neatly formatted block of metrics to standard output."""
        print(f"\n{'='*50}\n {title.upper()}\n{'='*50}")
        for k, v in metrics.items():
            print(f"{k:<32}: {v}")
        print(f"{'='*50}\n")
