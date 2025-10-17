import os
import argparse
from pathlib import Path
import orjson
import evaluation


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Centralized configuration for evaluation."""
    
    WORKSPACE_DIR = Path("dataset")
    BASELINE_NAME = "lmir_baseline"
    SPLITS = ["test", "val"]
    
    @classmethod
    def update_from_args(cls, args):
        """Update configuration from command line arguments."""
        if args.workspace:
            cls.WORKSPACE_DIR = Path(args.workspace)
        if args.baseline:
            cls.BASELINE_NAME = args.baseline
        if args.splits:
            cls.SPLITS = args.splits


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_split(baseline_name, split, workspace_dir):
    """
    Evaluate results for a single split.
    
    Args:
        baseline_name: Name of the baseline/model
        split: Split name ('test' or 'val')
        workspace_dir: Path to workspace directory
        
    Returns:
        dict: Dictionary of metric scores
    """
    results_dir = workspace_dir / "results" / baseline_name / split
    test_dir = workspace_dir / split
    
    if not results_dir.exists():
        print(f"Warning: Results directory not found: {results_dir}")
        return None
    
    if not test_dir.exists():
        print(f"Warning: Test directory not found: {test_dir}")
        return None
    
    # Get sorted file lists
    result_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.json')])
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.json')])
    
    if len(result_files) != len(test_files):
        print(f"Warning: Mismatch in file counts for {split} split")
        print(f"  Results: {len(result_files)}, Test: {len(test_files)}")
    
    # Add paths
    test_file_paths = [test_dir / f for f in test_files]
    result_file_paths = [results_dir / f for f in result_files]
    
    # Accumulate metrics dynamically
    metrics = {}
    metric_keys = None
    
    print(f"Evaluating {split} split ({len(test_files)} files)...")
    for test_file, result_file in zip(test_file_paths, result_file_paths):
        with open(test_file, "rb") as f:
            test_data = orjson.loads(f.read())
        with open(result_file, "rb") as f:
            result_data = orjson.loads(f.read())
        
        measures = evaluation.evaluation_report(result_data, test_data)
        
        # Initialize metrics dict on first file
        if metric_keys is None:
            metric_keys = list(measures.keys())
            for key in metric_keys:
                metrics[key] = 0
        
        # Accumulate metric values
        for key in metric_keys:
            if key in measures and "mean" in measures[key]:
                metrics[key] += measures[key]["mean"]
    
    # Calculate averages
    n = len(test_files)
    for metric in metrics:
        metrics[metric] /= n
    
    return metrics, metric_keys


def print_results(baseline_name, split_results, all_metric_keys):
    """
    Print evaluation results in a formatted table.
    
    Args:
        baseline_name: Name of the baseline/model
        split_results: Dictionary mapping split names to (metrics, keys) tuples
        all_metric_keys: List of all metric keys to display
    """
    print("\n" + "=" * (10 + 12 * len(all_metric_keys)))
    print(f"Evaluation Results for: {baseline_name}")
    print("=" * (10 + 12 * len(all_metric_keys)))
    
    # Print header
    header = f"{'Split':<10}"
    for key in all_metric_keys:
        header += f" {key:>11}"
    print(header)
    print("-" * (10 + 12 * len(all_metric_keys)))
    
    # Print results for each split
    for split, (metrics, _) in split_results.items():
        if metrics:
            row = f"{split:<10}"
            for key in all_metric_keys:
                value = metrics.get(key, 0)
                row += f" {value:>11.5f}"
            print(row)
    
    print("=" * (10 + 12 * len(all_metric_keys)) + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function."""
    split_results = {}
    all_metric_keys = []
    
    for split in Config.SPLITS:
        print(f"\n{'='*70}")
        print(f"Processing {split} split")
        print('='*70)
        
        result = evaluate_split(
            Config.BASELINE_NAME,
            split,
            Config.WORKSPACE_DIR
        )
        
        if result:
            metrics, metric_keys = result
            split_results[split] = (metrics, metric_keys)
            
            # Collect all unique metric keys
            for key in metric_keys:
                if key not in all_metric_keys:
                    all_metric_keys.append(key)
    
    # Print summary
    if split_results:
        print_results(Config.BASELINE_NAME, split_results, all_metric_keys)
    else:
        print("\nNo results to display. Check your paths and file names.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval results against ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--baseline",
        default="lmir_baseline",
        help="Name of the baseline/model to evaluate"
    )
    parser.add_argument(
        "--workspace",
        default="dataset",
        help="Path to workspace directory"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test", "val"],
        choices=["test", "val", "train"],
        help="Which splits to evaluate"
    )
    
    args = parser.parse_args()
    Config.update_from_args(args)
    
    main()
    