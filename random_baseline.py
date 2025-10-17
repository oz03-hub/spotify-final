import random
import argparse
from pathlib import Path
from tqdm import tqdm
from util import load_corpus, load_queries, get_query_files, save_results


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Centralized configuration for paths and parameters."""
    
    # Default paths
    WORKSPACE_DIR = Path("dataset")
    CORPUS_FILE = WORKSPACE_DIR / "track_corpus.json"
    QUERIES_DIR = WORKSPACE_DIR / "test"
    RESULTS_DIR = WORKSPACE_DIR / "results" / "random_baseline" / "test"
    
    # Model parameters
    NUM_RESULTS = 100  # Number of random results to return
    SEED = 42  # Random seed for reproducibility (None = random)
    
    @classmethod
    def update_from_args(cls, args):
        """Update configuration from command line arguments."""
        if args.workspace:
            cls.WORKSPACE_DIR = Path(args.workspace)
            cls.CORPUS_FILE = cls.WORKSPACE_DIR / "track_corpus.json"
        if args.corpus:
            cls.CORPUS_FILE = Path(args.corpus)
        if args.queries:
            cls.QUERIES_DIR = Path(args.queries)
        if args.results:
            cls.RESULTS_DIR = Path(args.results)
        if args.num_results:
            cls.NUM_RESULTS = args.num_results
        if args.seed is not None:
            cls.SEED = args.seed
        
        # Ensure results directory exists
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# RANDOM BASELINE
# ============================================================================
class RandomBaseline:
    """Random baseline that samples tracks uniformly."""
    
    def __init__(self, track_ids, seed=None):
        """
        Args:
            track_ids: List of all track IDs in corpus
            seed: Random seed for reproducibility
        """
        self.track_ids = track_ids
        if seed is not None:
            random.seed(seed)
    
    def retrieve(self, num_results):
        """
        Retrieve random tracks.
        
        Args:
            num_results: Number of results to return
            
        Returns:
            List of randomly sampled track IDs
        """
        return random.sample(self.track_ids, min(num_results, len(self.track_ids)))


# ============================================================================
# RESULTS PROCESSING
# ============================================================================
def process_queries(baseline, queries, corpus, num_results):
    """Process all queries and return results."""
    results = {}
    for playlist in tqdm(queries, desc="Sampling"):
        sampled_ids = baseline.retrieve(num_results)
        retrieved_list = []
        for track_id in sampled_ids:
            retrieved_list.append({
                "track_uri": corpus[track_id]["track_uri"],
                "artist_uri": corpus[track_id]["artist_uri"],
            })
        results[playlist["pid"]] = retrieved_list
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function."""
    # Load corpus
    print(f"Loading corpus from {Config.CORPUS_FILE}")
    corpus = load_corpus(Config.CORPUS_FILE)
    track_ids = list(corpus.keys())
    print(f"Loaded {len(track_ids)} tracks")
    
    # Initialize baseline
    baseline = RandomBaseline(track_ids, seed=Config.SEED)
    if Config.SEED is not None:
        print(f"Using random seed: {Config.SEED}")

    # Process all query files
    query_files = get_query_files(Config.QUERIES_DIR)
    print(f"\nFound {len(query_files)} query files")
    
    for query_file in query_files:
        print(f"\nProcessing: {query_file.name}")
        queries = load_queries(query_file)
        results = process_queries(baseline, queries, corpus, Config.NUM_RESULTS)
        save_results(results, query_file.name, Config.RESULTS_DIR)

    print("\n✓ All queries processed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Random Baseline for Music Track Retrieval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--workspace",
        help="Path to workspace directory"
    )
    parser.add_argument(
        "--corpus", 
        help="Path to corpus JSON file"
    )
    parser.add_argument(
        "--queries", 
        help="Path to directory containing query JSON files"
    )
    parser.add_argument(
        "--results",
        help="Path to output directory for results"
    )
    parser.add_argument(
        "--num_results", 
        type=int,
        help="Number of random results to return per query"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    Config.update_from_args(args)
    
    main()
