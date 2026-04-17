import time
import logging
import pandas as pd
from pathlib import Path
from contextlib import contextmanager

class MetricsTracker:
    """A utility class to track execution timing, log progress, and save metrics to CSV."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timing_data = {"Phase": [], "Duration (seconds)": [], "Timestamp": []}
        
    @contextmanager
    def log_and_time(self, name: str):
        """Context manager to log and time a block of code and record it in timing_data."""
        self.logger.info(f"Starting: {name}...")
        start = time.time()
        yield
        duration = time.time() - start
        
        self.timing_data["Phase"].append(name)
        self.timing_data["Duration (seconds)"].append(round(duration, 4))
        self.timing_data["Timestamp"].append(time.strftime('%Y-%m-%d %H:%M:%S'))

    def get_duration(self, phase_name: str) -> float:
        """Helper to fetch the duration of a specific timed phase."""
        try:
            idx = self.timing_data["Phase"].index(phase_name)
            return self.timing_data["Duration (seconds)"][idx]
        except ValueError:
            return None

    def flatten_eval_metrics(self, pipeline_name: str, eval_metrics: dict) -> dict:
        """Flattens nested evaluation metrics into a single-level dictionary and appends timestamps & runtimes."""
        flat_metrics = {"Timestamp": time.strftime('%Y-%m-%d %H:%M:%S'), "Pipeline": pipeline_name}
        
        for key, value in eval_metrics.items():
            if key == 'timing': continue
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    flat_metrics[f"{key}_{sub_key}"] = round(sub_val, 4) if isinstance(sub_val, float) else sub_val
            else:
                flat_metrics[key] = round(value, 4) if isinstance(value, float) else value
                
        # Automatically include total runtime if it was tracked
        total_runtime = self.get_duration('Total Pipeline Runtime')
        if total_runtime is not None:
            flat_metrics['Total Runtime (seconds)'] = total_runtime
            
        return flat_metrics

    def save_to_csv(self, data: dict, output_path: Path):
        """Saves dictionary data to a CSV file. Supports flat dictionaries and dictionaries of lists."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format: if all values are lists, it's already rectangular data
        if all(isinstance(v, list) for v in data.values()):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
            
        df.to_csv(output_path, mode='a', index=False, header=not output_path.exists())
        self.logger.info(f"Saved results to {output_path}")

    def print_timing_summary(self):
        """Logs a summary of all recorded timing phases."""
        self.logger.info("--- TIMING SUMMARY ---")
        for p, d in zip(self.timing_data["Phase"], self.timing_data["Duration (seconds)"]):
            self.logger.info(f"{p:<30}: {d} seconds")
            
    def print_metrics(self, title: str, metrics: dict):
        """Prints a neatly formatted block of metrics to standard output."""
        print(f"\n{'='*50}\n {title.upper()}\n{'='*50}")
        for k, v in metrics.items():
            print(f"{k:<32}: {v}")
        print(f"{'='*50}\n")
