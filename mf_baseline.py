import json
import argparse
import numpy as np
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import nltk
from scipy.sparse import csr_matrix, vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from util import load_corpus, load_queries, get_query_files, save_results, tokenize

# Ensure necessary tokenizer resources
nltk.download("punkt", quiet=True)


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Centralized configuration for paths and parameters."""

    # Default paths
    WORKSPACE_DIR = Path("dataset")
    CORPUS_FILE = WORKSPACE_DIR / "track_corpus.json"
    TRAIN_QUERIES_DIR = WORKSPACE_DIR / "train"
    RESULTS_DIR = WORKSPACE_DIR / "results" / "mf_baseline"

    # Model parameters
    N_COMPONENTS = 200  # Latent dimensions
    TOP_K = 100  # Number of results to return

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
        if args.n_components:
            cls.N_COMPONENTS = args.n_components
        if args.top_k:
            cls.TOP_K = args.top_k

        # Ensure results directory exists
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MATRIX FACTORIZATION MODEL
# ============================================================================
class MFModel:
    """Matrix Factorization model for playlist-track recommendation."""
    
    def __init__(self, corpus):
        """
        Initialize model with track corpus.
        
        Args:
            corpus: Dictionary mapping track_id to track metadata
        """
        self.corpus = corpus
        self.track_ids = list(corpus.keys())
        self.track_id_to_idx = {tid: idx for idx, tid in enumerate(self.track_ids)}
        
        # Build vocabulary from corpus
        print("Building vocabulary from corpus...")
        self.vocab = self._build_vocab_from_corpus()
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        print(f"Vocabulary size: {len(self.vocab)}")
        
        # Build track TF matrix
        print("Building track TF matrix...")
        self.track_tf_matrix = self._build_track_tf_matrix()
        print(f"Track TF matrix shape: {self.track_tf_matrix.shape}")
        
        # Placeholders for playlists and model
        self.playlist_tf_rows = []
        self.U = None
        self.V = None
        self.svd = None
        self.num_tracks = len(self.track_ids)

    def _build_vocab_from_corpus(self):
        """Extract all unique tokens from corpus extended_text fields."""
        vocab = set()
        for track_id, track_data in tqdm(self.corpus.items(), desc="Extracting vocab"):
            text = track_data.get("extended_text", "")
            tokens = tokenize(text)
            vocab.update(tokens)
        return sorted(list(vocab))

    def _build_track_tf_matrix(self):
        """Build sparse TF matrix for all tracks (rows=tracks, cols=vocab)."""
        rows = []
        cols = []
        data = []
        
        for track_idx, track_id in enumerate(tqdm(self.track_ids, desc="Building TF matrix")):
            track_data = self.corpus[track_id]
            text = track_data.get("extended_text", "")
            tokens = tokenize(text)
            tf_counter = Counter(tokens)
            
            for word, count in tf_counter.items():
                if word in self.word_to_idx:
                    rows.append(track_idx)
                    cols.append(self.word_to_idx[word])
                    data.append(count)
        
        return csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.track_ids), len(self.vocab)),
            dtype=np.float32
        )

    def add_playlist(self, playlist):
        """
        Add a playlist to the training data.
        
        Args:
            playlist: Dictionary with 'name' and 'tracks' fields
        """
        playlist_name = playlist.get("name", "")
        tokens = tokenize(playlist_name)
        tf_counter = Counter(tokens)
        
        # Build sparse row for this playlist
        cols = []
        data = []
        for word, count in tf_counter.items():
            if word in self.word_to_idx:
                cols.append(self.word_to_idx[word])
                data.append(count)
        
        if cols:  # Only add if playlist has valid tokens
            row = csr_matrix(
                (data, ([0] * len(cols), cols)),
                shape=(1, len(self.vocab)),
                dtype=np.float32
            )
            self.playlist_tf_rows.append(row)

    def compute_factorization(self, n_components=200):
        """
        Build combined TF matrix and perform SVD factorization.
        
        Args:
            n_components: Number of latent dimensions
        """
        print(f"\nBuilding combined matrix (tracks + playlists)...")
        
        # Stack playlists below tracks
        if self.playlist_tf_rows:
            playlist_tf_matrix = vstack(self.playlist_tf_rows)
            print(f"Playlist TF matrix shape: {playlist_tf_matrix.shape}")
            combined_matrix = vstack([self.track_tf_matrix, playlist_tf_matrix])
        else:
            print("Warning: No playlists added to training data")
            combined_matrix = self.track_tf_matrix
        
        print(f"Combined matrix shape: {combined_matrix.shape}")
        print(f"Matrix density: {combined_matrix.nnz / (combined_matrix.shape[0] * combined_matrix.shape[1]):.6f}")
        
        # Perform SVD
        print(f"\nRunning Truncated SVD (n_components={n_components})...")
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        U = self.svd.fit_transform(combined_matrix)
        V = self.svd.components_.T
        
        # Normalize embeddings
        self.U = normalize(U, norm='l2', axis=1)
        self.V = normalize(V, norm='l2', axis=1)
        
        print(f"Factorization complete:")
        print(f"  U shape: {self.U.shape}")
        print(f"  V shape: {self.V.shape}")
        print(f"  Explained variance ratio: {self.svd.explained_variance_ratio_.sum():.4f}")

    def retrieve(self, query_text, top_k=100):
        """
        Retrieve top-k tracks for a query.
        
        Args:
            query_text: Query string (e.g., playlist name + description)
            top_k: Number of results to return
            
        Returns:
            List of (track_id, score) tuples
        """
        if self.U is None or self.V is None:
            raise ValueError("Model not trained. Call compute_factorization() first.")
        
        # Tokenize and build query vector
        tokens = tokenize(query_text)
        tf_counter = Counter(tokens)
        
        # Build sparse query vector (only in-vocab words)
        cols = []
        data = []
        for word, count in tf_counter.items():
            if word in self.word_to_idx:
                cols.append(self.word_to_idx[word])
                data.append(count)
        
        if not cols:
            # No valid tokens, return empty results
            return []
        
        query_vec = csr_matrix(
            (data, ([0] * len(cols), cols)),
            shape=(1, len(self.vocab)),
            dtype=np.float32
        )
        
        # Project query into latent space
        query_emb = query_vec.dot(self.V)
        query_emb = normalize(query_emb, norm='l2', axis=1)
        
        # Compute similarities with track embeddings only
        track_embeddings = self.U[:self.num_tracks]
        similarities = track_embeddings.dot(query_emb.T).ravel()
        
        # Get top-k tracks
        top_k = min(top_k, len(similarities))
        top_indices = np.argpartition(-similarities, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]
        
        results = [(self.track_ids[idx], float(similarities[idx])) for idx in top_indices]
        return results


# ============================================================================
# RESULTS PROCESSING
# ============================================================================
def process_queries(model, queries, corpus, top_k):
    """Process all queries and return results."""
    results = {}
    for playlist in tqdm(queries, desc="Retrieving"):
        query_text = f"{playlist.get('name', '')} {playlist.get('description', '')}"
        pid = playlist.get("pid")
        retrieved = model.retrieve(query_text, top_k=top_k)
        track_results = []
        for track_id, _ in retrieved:
            track = {
                "track_uri": corpus[track_id]["track_uri"],
                "artist_uri": corpus[track_id]["artist_uri"],
            }
            track_results.append(track)
        results[pid] = track_results
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function."""
    # Load corpus
    print(f"Loading corpus from {Config.CORPUS_FILE}")
    corpus = load_corpus(Config.CORPUS_FILE)
    
    # Initialize model
    model = MFModel(corpus)
    
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
    model.compute_factorization(n_components=Config.N_COMPONENTS)
    
    # Process test and val splits
    for split in ["test", "val"]:
        split_dir = Config.WORKSPACE_DIR / split
        if not split_dir.exists():
            print(f"\nSkipping {split} split (directory not found)")
            continue
        
        split_result_dir = Config.RESULTS_DIR / split
        split_result_dir.mkdir(parents=True, exist_ok=True)
        
        query_files = get_query_files(split_dir)
        print(f"\n{'='*70}")
        print(f"Processing {split} split ({len(query_files)} files)")
        print('='*70)
        
        for query_file in query_files:
            print(f"\nProcessing: {query_file.name}")
            queries = load_queries(query_file)
            results = process_queries(model, queries, corpus, Config.TOP_K)
            save_results(results, query_file.name, split_result_dir)
    
    print("\n✓ All queries processed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Matrix Factorization Baseline for Music Track Retrieval",
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
        "--train_queries",
        help="Path to directory containing training query JSON files"
    )
    parser.add_argument(
        "--results",
        help="Path to output directory for results"
    )
    parser.add_argument(
        "--n_components",
        type=int,
        help="Number of latent dimensions for SVD"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        help="Number of top results to return per query"
    )
    
    args = parser.parse_args()
    Config.update_from_args(args)
    
    main()