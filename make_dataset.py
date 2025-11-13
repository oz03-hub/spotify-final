import random
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import orjson
from util import (
    tokenize,
    load_corpus,
    load_queries,
    get_query_files,
    save_results,
    extract_id,
)
from lmir_baseline import DirichletLMRetriever, process_queries
from bm25_baseline import BM25Retriever
from svd_baseline import SVDModel
from wmf_baseline import WMFModel


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Centralized configuration for paths and parameters."""

    # Default paths
    WORKSPACE_DIR = Path("dataset")
    INVERTED_INDEX = WORKSPACE_DIR / "inverted_index.json"
    CORPUS_FILE = WORKSPACE_DIR / "track_corpus.json"

    TRAIN_QUERIES_DIR = WORKSPACE_DIR / "train"
    OUTPUT_DIR = WORKSPACE_DIR / "reranker_dataset" / "train"

    TEST_QUERIES_DIR = WORKSPACE_DIR / "test"
    TEST_OUTPUT_DIR = WORKSPACE_DIR / "reranker_dataset" / "test"

    VAL_QUERIES_DIR = WORKSPACE_DIR / "val"
    VAL_OUTPUT_DIR = WORKSPACE_DIR / "reranker_dataset" / "val"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Model parameters
    MU = 2000  # Dirichlet smoothing parameter
    K1 = 1.2
    B = 0.75
    N_COMPONENTS = 200

    REGULARIZATION = 0.01
    ITERATIONS = 50
    ALPHA = 30.0

    TOP_K = 1000  # Number of top results to return from first stage
    MIN_NON_RELEVANT = 6  # Minimum number of non-relevant samples

    @classmethod
    def update_from_args(cls, args):
        """Update configuration from command line arguments."""
        if args.corpus:
            cls.CORPUS_FILE = Path(args.corpus)
        if args.queries:
            cls.TRAIN_QUERIES_DIR = Path(args.queries)
        if args.output:
            cls.OUTPUT_DIR = Path(args.output)
        if args.mu:
            cls.MU = args.mu
        if args.top_k:
            cls.TOP_K = args.top_k
        if args.ce_batch_size:
            cls.CROSS_ENCODER_BATCH_SIZE = args.ce_batch_size
        if args.min_non_relevant:
            cls.MIN_NON_RELEVANT = args.min_non_relevant

        # Ensure results directory exists
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sample_tracks(retrieved_tracks, relevant_track_ids):
    """
    Sample tracks to create balanced training data.
    Keep all relevant tracks and sample non-relevant tracks.

    Args:
        retrieved_tracks: List of (track_id, score) tuples
        relevant_track_ids: Set of relevant track IDs

    Returns:
        List of sampled (track_id, score) tuples
    """
    relevant = []
    non_relevant = []

    for track_id, score in retrieved_tracks:
        if track_id in relevant_track_ids:
            relevant.append((track_id, score))
        else:
            non_relevant.append((track_id, score))

    num_relevant = len(relevant)
    num_non_relevant_to_sample = max(num_relevant, Config.MIN_NON_RELEVANT)

    # Sample non-relevant tracks
    if len(non_relevant) > num_non_relevant_to_sample:
        sampled_non_relevant = random.sample(non_relevant, num_non_relevant_to_sample)
    else:
        sampled_non_relevant = non_relevant

    # Combine and return
    return relevant + sampled_non_relevant


def compute_features(
    query_info,
    tracks,
    corpus,
    bm25: BM25Retriever,
    mf: SVDModel,
    wmf: WMFModel,
    relevant_track_ids,
):
    """
    Compute all features for tracks of a single playlist.
    Returns feature dictionaries.
    """
    query_text = f"{query_info.get('name', '')} {query_info.get('description', '')}"
    query_tokens = tokenize(query_text)

    features = []

    track_ids = [track[0] for track in tracks]
    mf_scores = mf.score_query_tracks(query_text, track_ids)
    wmf_scores = wmf.score_query_tracks(query_text, track_ids)

    for i, (track_id, dirichlet_score) in enumerate(tracks):
        track_text = corpus[track_id]["extended_text"]
        track_tokens = tokenize(track_text)
        track_tf = Counter(track_tokens)

        # Compute features
        bm25_score = bm25.score_query_doc(query_text, track_text)
        tf_score = sum(track_tf[token] for token in query_tokens)
        dl_score = sum(track_tf.values())
        pf_score = corpus[track_id]["pf"]
        duration = corpus[track_id].get("duration_ms", 0)
        # mf_score = float(mf.score_query_doc(query_text, track_id))
        # wmf_score = float(wmf.score_query_doc(query_text, track_id))
        label = 1 if track_id in relevant_track_ids else 0

        features.append(
            {
                "track_id": track_id,
                "diriclet_score": dirichlet_score,
                "bm25_score": bm25_score,
                "tf_score": tf_score,
                "dl_score": dl_score,
                "duration_score": duration,
                "mf_score": mf_scores[i],
                "pf_score": pf_score,
                "wmf_score": wmf_scores[i],
                "label": label,
            }
        )

    return features


def process_query_set(
    queries_dir,
    output_dir,
    dirichlet,
    bm25,
    mf,
    wmf,
    corpus,
    dataset_type="train",
):
    """
    Process a set of queries (train/test/val).

    Args:
        queries_dir: Directory containing query files
        output_dir: Directory to save results
        dirichlet: Dirichlet retriever
        bm25: BM25 retriever
        mf: Matrix factorization model
        wmf: Weighted matrix factorization model
        corpus: Track corpus
        dataset_type: 'train', 'test', or 'val'
    """
    print(f"\n{'=' * 80}")
    print(f"Processing {dataset_type.upper()} set")
    print(f"{'=' * 80}")

    for query_file in get_query_files(queries_dir):
        print(f"\n{'=' * 80}")
        print(f"Processing: {query_file.name}")
        print(f"{'=' * 80}")

        queries = load_queries(query_file)
        relevant_tracks = {
            p_info["pid"]: set(extract_id(t["track_uri"]) for t in p_info["tracks"])
            for p_info in queries
        }

        # 1st stage retrieval
        print("Running first-stage retrieval...")
        results = process_queries(dirichlet, queries, Config.TOP_K)

        # Process each playlist
        final_results = {}
        print("Computing features...")
        for idx, query_info in enumerate(tqdm(queries, desc="Processing playlists")):
            pid = query_info["pid"]

            # For training: sample tracks; for test/val: use all retrieved tracks
            if dataset_type == "train":
                tracks = sample_tracks(results[pid], relevant_tracks[pid])
            else:
                tracks = results[pid]

            # Compute features
            features = compute_features(
                query_info,
                tracks,
                corpus,
                bm25,
                mf,
                wmf,
                relevant_tracks[pid],
            )

            final_results[pid] = features

        # Save results
        print("\nSaving results...")
        save_results(final_results, query_file.name, output_dir)

        print(f"Completed {query_file.name}")


def main():
    print("Loading inverted index...")
    with open(Config.INVERTED_INDEX, "r") as f:
        inverted_index = orjson.loads(f.read())

    print("Loading corpus...")
    corpus = load_corpus(Config.CORPUS_FILE)

    print("Initializing retrievers...")
    dirichlet = DirichletLMRetriever(mu=Config.MU)
    dirichlet.load_inverted_index_from_object(inverted_index)

    bm25 = BM25Retriever(k1=Config.K1, b=Config.B)
    bm25.load_inverted_index_from_object(inverted_index)

    print("Initializing matrix factorization model...")
    mf = SVDModel(corpus)
    train_query_files = get_query_files(Config.TRAIN_QUERIES_DIR)
    print(f"\nFound {len(train_query_files)} training query files")

    print("Initializing weighted matrix factorization model...")
    wmf = WMFModel.load(model_dir="model/wmf", corpus=corpus)

    # Train MF model on training data only
    for query_file in train_query_files:
        print(f"\nProcessing training queries from: {query_file.name}")
        queries = load_queries(query_file)
        for query in tqdm(queries, desc="Adding playlists"):
            # wmf.add_playlist(query)
            mf.add_playlist(query)

    print(f"\nTraining model with {len(mf.playlist_tf_rows)} playlists...")
    mf.compute_factorization(n_components=Config.N_COMPONENTS)

    # Process training set (with sampling)
    process_query_set(
        Config.TRAIN_QUERIES_DIR,
        Config.OUTPUT_DIR,
        dirichlet,
        bm25,
        mf,
        wmf,
        corpus,
        dataset_type="train",
    )

    process_query_set(
        Config.TEST_QUERIES_DIR,
        Config.TEST_OUTPUT_DIR,
        dirichlet,
        bm25,
        mf,
        wmf,
        corpus,
        dataset_type="test",
    )

    process_query_set(
        Config.VAL_QUERIES_DIR,
        Config.VAL_OUTPUT_DIR,
        dirichlet,
        bm25,
        mf,
        wmf,
        corpus,
        dataset_type="val",
    )

    print("\n" + "=" * 80)
    print("All queries processed successfully")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simplified Feature Extraction with Balanced Sampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--corpus", help="Path to corpus JSON file")
    parser.add_argument(
        "--queries", help="Path to directory containing query JSON files"
    )
    parser.add_argument("--output", help="Path to output directory for results")
    parser.add_argument(
        "--mu",
        type=float,
        help="Dirichlet smoothing parameter (larger = more smoothing)",
    )
    parser.add_argument(
        "--top_k", type=int, help="Number of top results to return per query"
    )
    parser.add_argument(
        "--ce_batch_size", type=int, help="Batch size for cross-encoder processing"
    )
    parser.add_argument(
        "--min_non_relevant",
        type=int,
        help="Minimum number of non-relevant samples per playlist",
    )

    args = parser.parse_args()
    Config.update_from_args(args)

    main()
