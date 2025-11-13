import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
import nltk
from util import tokenize, load_corpus, load_queries, get_query_files, save_results

# Ensure necessary tokenizer resources
nltk.download("punkt", quiet=True)


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Centralized configuration for paths and parameters."""

    # Default paths
    WORKSPACE_DIR = Path("dataset")
    INVERTED_INDEX = WORKSPACE_DIR / "inverted_index.json"
    CORPUS_FILE = WORKSPACE_DIR / "track_corpus.json"
    TEST_QUERIES_DIR = WORKSPACE_DIR / "test"
    VAL_QUERIES_DIR = WORKSPACE_DIR / "val"

    RESULTS_DIR = WORKSPACE_DIR / "results" / "lmir_baseline"
    TEST_RESULTS_DIR = RESULTS_DIR / "test"
    VAL_RESULTS_DIR = RESULTS_DIR / "val"

    # Model parameters
    MU = 2000  # Dirichlet smoothing parameter
    TOP_K = 1000  # Number of top results to return

    @classmethod
    def update_from_args(cls, args):
        """Update configuration from command line arguments."""
        if args.corpus:
            cls.CORPUS_FILE = Path(args.corpus)
        if args.queries:
            cls.QUERIES_DIR = Path(args.queries)
        if args.results:
            cls.RESULTS_DIR = Path(args.results)
        if args.inverted_index:
            cls.INVERTED_INDEX = Path(args.inverted_index)
        if args.mu:
            cls.MU = args.mu
        if args.top_k:
            cls.TOP_K = args.top_k

        # Ensure results directory exists
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.VAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# RETRIEVAL MODEL
# ============================================================================
class DirichletLMRetriever:
    """Dirichlet Language Model retriever using inverted index."""

    def __init__(self, mu=2000):
        """
        Args:
            mu: Dirichlet smoothing parameter. Larger values = more smoothing.
        """
        self.mu = mu
        self.inverted_index = {}
        self.doc_lens = {}
        self.collection_freq = Counter()
        self.collection_len = 0
        self.total_docs = 0
    
    def load_inverted_index_from_object(self, data):
        self.inverted_index = data
        self.doc_lens = data["DOC_LENGTHS"]
        self.total_docs = data["TOTAL_DOCS"]

        # Build collection frequency and collection length
        for term, postings in self.inverted_index.items():
            if term in ["DOCID_MAP", "DOC_LENGTHS", "TOTAL_DOCS"]:
                continue

            cf = sum(tf for _, tf in postings)
            self.collection_freq[term] = cf
            self.collection_len += cf


    def load_inverted_index(self, index_path):
        """
        Load inverted index from JSON file.

        Args:
            index_path: Path to inverted index JSON file
        """
        print(f"Loading inverted index from {index_path}")
        with open(index_path, "r") as f:
            data = json.load(f)
        self.load_inverted_index_from_object(data)

    def retrieve(self, query_text, top_k=10):
        """
        Retrieve and rank documents for a query using Dirichlet prior smoothing.

        Args:
            query_text: Query string
            top_k: Number of top results to return

        Returns:
            List of (doc_id, score) tuples sorted by descending score.
        """
        query_tokens = tokenize(query_text)
        scores = defaultdict(float)

        P_tc = {t: self.collection_freq[t] / self.collection_len for t in query_tokens if t in self.collection_freq}

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            p_tc = P_tc.get(term, 1e-10)
            postings = self.inverted_index[term]

            for doc_id, f_td in postings:
                track_id = self.inverted_index["DOCID_MAP"][doc_id]
                doc_len = self.doc_lens[doc_id]
                p_td = (f_td + self.mu * p_tc) / (doc_len + self.mu)
                scores[track_id] += math.log(p_td)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ============================================================================
# RESULTS PROCESSING
# ============================================================================
def process_queries(retriever, queries, top_k):
    """Process all queries and return results."""
    results = {}
    for playlist in tqdm(queries, desc="Retrieving"):
        query_text = f"{playlist.get('name', '')} {playlist.get('description', '')}"
        pid = playlist.get("pid")
        results[pid] = retriever.retrieve(query_text, top_k=top_k)
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function."""
    # Load inverted index
    retriever = DirichletLMRetriever(mu=Config.MU)
    retriever.load_inverted_index(Config.INVERTED_INDEX)

    corpus = load_corpus(Config.CORPUS_FILE)

    for query_dir, result_dir in [
        (Config.TEST_QUERIES_DIR, Config.TEST_RESULTS_DIR),
        (Config.VAL_QUERIES_DIR, Config.VAL_RESULTS_DIR),
    ]:
        print(f"\nProcessing queries in directory: {query_dir}")
        query_files = get_query_files(query_dir)
        print(f"\nFound {len(query_files)} query files")

        for query_file in query_files:
            print(f"\nProcessing: {query_file.name}")
            queries = load_queries(query_file)
            results = process_queries(retriever, queries, Config.TOP_K)

            # Map track_ids to track objects
            for pid in results:
                replacement_tracks = []
                for track_id, _ in results[pid]:
                    track = corpus[track_id]
                    eval_track = {
                        "track_uri": track["track_uri"],
                        "artist_uri": track["artist_uri"],
                    }
                    replacement_tracks.append(eval_track)
                results[pid] = replacement_tracks

            save_results(results, query_file.name, result_dir)
    print("\n✓ All queries processed successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Dirichlet Language Model Retriever for Music Track Corpus (with Inverted Index)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--corpus", help="Path to corpus JSON file")
    parser.add_argument(
        "--queries", help="Path to directory containing query JSON files"
    )
    parser.add_argument("--results", help="Path to output directory for results")
    parser.add_argument("--inverted_index", help="Path to inverted index JSON file")
    parser.add_argument(
        "--mu",
        type=float,
        help="Dirichlet smoothing parameter (larger = more smoothing)",
    )
    parser.add_argument(
        "--top_k", type=int, help="Number of top results to return per query"
    )

    args = parser.parse_args()
    Config.update_from_args(args)

    main()
