import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
from pathlib import Path
from tqdm import tqdm
from util import load_corpus, load_queries, get_query_files, save_results
from wmf_baseline import WMFModel

'''
Sourced from wmf_baseline.py, modified to retrieve top-k similar playlists given an inputted playlist name
'''

class Config:
    """Centralized configuration for paths and parameters."""

    # Default paths
    WORKSPACE_DIR = Path("dataset")
    CORPUS_FILE = WORKSPACE_DIR / "track_corpus.json"
    TRAIN_QUERIES_DIR = WORKSPACE_DIR / "train"
    RESULTS_DIR = WORKSPACE_DIR / "results" / "wmf_playlists"
    MODEL_DIR = Path("model")

    # Model parameters
    N_FACTORS = 200  # Latent factors
    REGULARIZATION = 0.01  # L2 regularization
    ITERATIONS = 50  # Number of ALS iterations
    ALPHA = 30.0  # Confidence weight multiplier
    TOP_K = 1000  # Number of results to return

    # Training options
    RETRAIN = False  # Whether to retrain even if model exists

    @classmethod
    def update_from_args(cls, args):
        """Update configuration from command line arguments."""
        if args.workspace:
            cls.WORKSPACE_DIR = Path(args.workspace)
            cls.CORPUS_FILE = cls.WORKSPACE_DIR / "track_corpus.json"
            cls.TRAIN_QUERIES_DIR = cls.WORKSPACE_DIR / "train"
        if args.corpus:
            cls.CORPUS_FILE = Path(args.corpus)
        if args.train_queries:
            cls.TRAIN_QUERIES_DIR = Path(args.train_queries)
        if args.results:
            cls.RESULTS_DIR = Path(args.results)
        if args.model_dir:
            cls.MODEL_DIR = Path(args.model_dir)
        if args.n_factors:
            cls.N_FACTORS = args.n_factors
        if args.regularization:
            cls.REGULARIZATION = args.regularization
        if args.iterations:
            cls.ITERATIONS = args.iterations
        if args.alpha:
            cls.ALPHA = args.alpha
        if args.top_k:
            cls.TOP_K = args.top_k
        if args.retrain:
            cls.RETRAIN = True

        # Ensure directories exist
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)

def process_queries(model: WMFModel, queries, top_k):
    """Process all queries and return results."""
    results = {}
    for playlist in tqdm(queries, desc="Retrieving"):
        query_text = f"{playlist.get('name', '')} {playlist.get('description', '')}"
        pid = playlist.get("pid")
        # retrieved = [(playlist_id, score), ...]
        retrieved = model.retrieve_playlists(query_text, top_k=top_k)
        results[pid] = retrieved
    # results = "query playlist": [(playlist_id, score), ...] sorted in descending order
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function."""
    # Load corpus
    print(f"Loading corpus from {Config.CORPUS_FILE}")
    corpus = load_corpus(Config.CORPUS_FILE)

    # Define model path
    model_path = Config.MODEL_DIR / "wmf"
    model_exists = (model_path / "metadata.pkl").exists()

    # Determine whether to train or load
    should_train = Config.RETRAIN or not model_exists

    if should_train:
        print(f"\n{'=' * 70}")
        print("TRAINING NEW MODEL")
        print("=" * 70)

        # Initialize model
        model = WMFModel(corpus)

        # Load training playlists
        train_query_files = get_query_files(Config.TRAIN_QUERIES_DIR)
        print(f"\nFound {len(train_query_files)} training query files")

        for query_file in train_query_files:
            print(f"\nProcessing training queries from: {query_file.name}")
            queries = load_queries(query_file)
            for query in tqdm(queries, desc="Adding playlists"):
                model.add_playlist(query)

        # Train model
        print(f"\nTraining model with {len(model.playlist_tf_rows)} playlists...")
        model.compute_factorization(
            n_factors=Config.N_FACTORS,
            regularization=Config.REGULARIZATION,
            iterations=Config.ITERATIONS,
            alpha=Config.ALPHA,
        )

        # Save model
        model.save(model_path)
    else:
        print(f"\n{'=' * 70}")
        print("LOADING EXISTING MODEL")
        print("=" * 70)

        # Load existing model
        model = WMFModel.load(model_path, corpus)

    # Process test and val splits
    for split in ["test", "val"]:
        split_dir = Config.WORKSPACE_DIR / split
        if not split_dir.exists():
            print(f"\nSkipping {split} split (directory not found)")
            continue

        split_result_dir = Config.RESULTS_DIR / split
        split_result_dir.mkdir(parents=True, exist_ok=True)

        query_files = get_query_files(split_dir)
        print(f"\n{'=' * 70}")
        print(f"Processing {split} split ({len(query_files)} files)")
        print("=" * 70)

        for query_file in query_files:
            print(f"\nProcessing: {query_file.name}")
            queries = load_queries(query_file)
            results = process_queries(model, queries, Config.TOP_K)
            save_results(results, query_file.name, split_result_dir)

    print("\nAll queries processed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Weighted Matrix Factorization Baseline for Playlist Retrieval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace", help="Path to workspace directory")
    parser.add_argument("--corpus", help="Path to corpus JSON file")
    parser.add_argument(
        "--train_queries", help="Path to directory containing training query JSON files"
    )
    parser.add_argument("--results", help="Path to output directory for results")
    parser.add_argument(
        "--model_dir", help="Path to directory for saving/loading models"
    )
    parser.add_argument(
        "--n_factors", type=int, help="Number of latent factors for WMF"
    )
    parser.add_argument(
        "--regularization", type=float, help="L2 regularization parameter"
    )
    parser.add_argument("--iterations", type=int, help="Number of ALS iterations")
    parser.add_argument(
        "--alpha", type=float, help="Confidence weight multiplier (C = 1 + alpha * r)"
    )
    parser.add_argument(
        "--top_k", type=int, help="Number of top results to return per query"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining even if saved model exists",
    )

    args = parser.parse_args()
    Config.update_from_args(args)

    main()