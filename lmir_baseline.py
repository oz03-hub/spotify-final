import json
import math
import os
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import nltk

# Ensure necessary tokenizer resources
nltk.download('punkt', quiet=True)


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Centralized configuration for paths and parameters."""
    
    # Default paths
    WORKSPACE_DIR = Path("dataset")
    CORPUS_FILE = WORKSPACE_DIR / "tracks_index.json"
    QUERIES_DIR = WORKSPACE_DIR / "test"
    RESULTS_DIR = WORKSPACE_DIR / "results" / "lmir_baseline"
    
    # Model parameters
    MU = 2000  # Dirichlet smoothing parameter
    TOP_K = 100  # Number of top results to return
    
    @classmethod
    def update_from_args(cls, args):
        """Update configuration from command line arguments."""
        if args.corpus:
            cls.CORPUS_FILE = Path(args.corpus)
        if args.queries:
            cls.QUERIES_DIR = Path(args.queries)
        if args.results:
            cls.RESULTS_DIR = Path(args.results)
        if args.mu:
            cls.MU = args.mu
        if args.top_k:
            cls.TOP_K = args.top_k
        
        # Ensure results directory exists
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# TEXT PROCESSING
# ============================================================================
def tokenize(text):
    """Simple word tokenizer and normalizer."""
    return [w.lower() for w in nltk.word_tokenize(text) if w.isalnum()]


# ============================================================================
# RETRIEVAL MODEL
# ============================================================================
class DirichletLMRetriever:
    """Dirichlet Language Model retriever for document ranking."""
    
    def __init__(self, mu=2000):
        """
        Args:
            mu: Dirichlet smoothing parameter. Larger values = more smoothing.
        """
        self.mu = mu
        self.collection_freq = Counter()
        self.collection_len = 0
        self.doc_lens = {}
        self.doc_freqs = {}

    def build_index(self, corpus):
        """
        Build term statistics for all documents.
        
        Args:
            corpus: dict {docid: {"artist": str, "track": str, "album": str}}
        """
        for docid, data in tqdm(corpus.items(), desc="Indexing corpus"):
            text = f"{data.get('artist', '')} {data.get('track', '')} {data.get('album', '')}"
            tokens = tokenize(text)
            term_freqs = Counter(tokens)
            
            self.doc_freqs[docid] = term_freqs
            self.doc_lens[docid] = sum(term_freqs.values())
            self.collection_freq.update(term_freqs)
            self.collection_len += sum(term_freqs.values())

    def score(self, query_tokens, docid):
        """
        Compute Dirichlet-smoothed log-likelihood score.
        
        Args:
            query_tokens: List of tokenized query terms
            docid: Document identifier
            
        Returns:
            float: Log-likelihood score
        """
        score = 0.0
        doc_freqs = self.doc_freqs[docid]
        doc_len = self.doc_lens[docid]

        for term in query_tokens:
            f_qi_D = doc_freqs.get(term, 0)
            p_qi_C = (self.collection_freq.get(term, 0) / self.collection_len 
                     if self.collection_len > 0 else 0)
            term_prob = (f_qi_D + self.mu * p_qi_C) / (doc_len + self.mu)
            
            if term_prob > 0:
                score += math.log(term_prob)
                
        return score

    def retrieve(self, query_text, top_k=10):
        """
        Retrieve and rank documents for a query.
        
        Args:
            query_text: Query string
            top_k: Number of top results to return
            
        Returns:
            List of (docid, score) tuples, sorted by score descending
        """
        query_tokens = tokenize(query_text)
        scores = [(docid, self.score(query_tokens, docid)) 
                 for docid in self.doc_freqs]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ============================================================================
# DATA LOADING
# ============================================================================
def load_corpus(corpus_file):
    """Load corpus from JSON file."""
    print(f"Loading corpus from: {corpus_file}")
    with open(corpus_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_queries(query_file):
    """Load queries from JSON file."""
    with open(query_file, "r", encoding="utf-8") as f:
        return json.load(f)["playlists"]


def get_query_files(queries_dir):
    """Get all query files from directory."""
    queries_path = Path(queries_dir)
    return [queries_path / f for f in os.listdir(queries_path) 
            if f.endswith('.json')]


# ============================================================================
# RESULTS PROCESSING
# ============================================================================
def process_queries(retriever, queries, top_k):
    """Process all queries and return results."""
    results = {}
    for playlist in tqdm(queries, desc="Retrieving"):
        query_text = f"{playlist.get('name', '')} {playlist.get('description', '')}"
        results[playlist.get("pid")] = retriever.retrieve(query_text, top_k=top_k)
    return results


def save_results(results, query_file, results_dir):
    """Save results to JSON file."""
    # Create output filename preserving original query filename
    output_file = results_dir / query_file
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results written to: {output_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function."""
    # Load corpus and build index
    corpus = load_corpus(Config.CORPUS_FILE)
    retriever = DirichletLMRetriever(mu=Config.MU)
    retriever.build_index(corpus)

    # Process all query files
    query_files = get_query_files(Config.QUERIES_DIR)
    print(f"Found {len(query_files)} query files")
    
    for query_file in query_files:
        print(f"\nProcessing: {query_file.name}")
        queries = load_queries(query_file)
        results = process_queries(retriever, queries, Config.TOP_K)
        for pid in results:
            results[pid] = [corpus[docid] for docid, _ in results[pid]]
        save_results(results, query_file.name, Config.RESULTS_DIR)

    print("\n✓ All queries processed successfully")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Dirichlet Language Model Retriever for Music Track Corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--corpus", 
        help="Path to corpus JSON file",
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
        "--mu", 
        type=float,
        help="Dirichlet smoothing parameter (larger = more smoothing)"
    )
    parser.add_argument(
        "--top_k", 
        type=int,
        help="Number of top results to return per query"
    )
    
    args = parser.parse_args()
    Config.update_from_args(args)
    
    main()