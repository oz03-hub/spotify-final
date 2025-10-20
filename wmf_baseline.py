import json
import argparse
import numpy as np
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import nltk
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz
from implicit.als import AlternatingLeastSquares
import pickle

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
    RESULTS_DIR = WORKSPACE_DIR / "results" / "wmf_baseline"
    MODEL_DIR = Path("model")

    # Model parameters
    N_FACTORS = 200  # Latent factors
    REGULARIZATION = 0.01  # L2 regularization
    ITERATIONS = 15  # Number of ALS iterations
    ALPHA = 40.0  # Confidence weight multiplier
    TOP_K = 100  # Number of results to return
    
    # Training options
    RETRAIN = False  # Whether to retrain even if model exists
    USE_GPU = False  # Whether to use GPU if available

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
        if args.no_gpu:
            cls.USE_GPU = False

        # Ensure directories exist
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# WEIGHTED MATRIX FACTORIZATION MODEL
# ============================================================================
class WMFModel:
    """Weighted Matrix Factorization model for playlist-track recommendation."""
    
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
        self.model = None
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

    def compute_factorization(self, n_factors=200, regularization=0.01, 
                            iterations=15, alpha=40.0, use_gpu=True):
        """
        Build combined TF matrix and perform WMF using ALS.
        
        Args:
            n_factors: Number of latent factors
            regularization: L2 regularization parameter
            iterations: Number of ALS iterations
            alpha: Confidence weight multiplier (C = 1 + alpha * r)
            use_gpu: Whether to use GPU if available
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
        
        # Convert to item-user format for implicit library
        interaction_matrix = combined_matrix.tocsr().astype(np.float32)
        
        print(f"\nTraining Weighted Matrix Factorization:")
        print(f"  Factors: {n_factors}")
        print(f"  Regularization: {regularization}")
        print(f"  Iterations: {iterations}")
        print(f"  Alpha: {alpha}")
        print(f"  GPU: {use_gpu}")
        
        # Initialize and train WMF model
        self.model = AlternatingLeastSquares(
            factors=n_factors,
            regularization=regularization,
            iterations=iterations,
            alpha=alpha,
            use_gpu=use_gpu,
            random_state=42,
            calculate_training_loss=True
        )
        
        self.model.fit(interaction_matrix, show_progress=True)
        
        print(f"\nFactorization complete:")
        print(f"  Item factors shape: {self.model.item_factors.shape}")
        print(f"  User factors shape: {self.model.user_factors.shape}")

    def retrieve(self, query_text, top_k=100):
        """
        Retrieve top-k tracks for a query.
        
        Args:
            query_text: Query string (e.g., playlist name + description)
            top_k: Number of results to return
            
        Returns:
            List of (track_id, score) tuples
        """
        if self.model is None:
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
        
        # Get item factors (vocab embeddings) and user factors (entity embeddings)
        item_factors = self.model.item_factors  # vocab embeddings
        user_factors = self.model.user_factors  # entity embeddings (tracks + playlists)
        
        # Project query into latent space using item factors
        query_emb = query_vec.dot(item_factors).ravel()
        
        # Compute similarities with track embeddings only
        track_embeddings = user_factors[:self.num_tracks]
        similarities = track_embeddings.dot(query_emb)
        
        # Get top-k tracks
        top_k = min(top_k, len(similarities))
        top_indices = np.argpartition(-similarities, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]
        
        results = [(self.track_ids[idx], float(similarities[idx])) for idx in top_indices]
        return results

    def save(self, model_dir):
        """
        Save model to disk.
        
        Args:
            model_dir: Directory to save model files
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving model to {model_dir}/")
        
        # Save metadata
        metadata = {
            'track_ids': self.track_ids,
            'track_id_to_idx': self.track_id_to_idx,
            'vocab': self.vocab,
            'word_to_idx': self.word_to_idx,
            'num_tracks': self.num_tracks,
        }
        with open(model_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save sparse matrices
        save_npz(model_dir / 'track_tf_matrix.npz', self.track_tf_matrix)
        if self.playlist_tf_rows:
            playlist_tf_matrix = vstack(self.playlist_tf_rows)
            save_npz(model_dir / 'playlist_tf_matrix.npz', playlist_tf_matrix)
        
        # Save ALS model
        if self.model is not None:
            with open(model_dir / 'als_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
        
        print("✓ Model saved successfully")

    @classmethod
    def load(cls, model_dir, corpus):
        """
        Load model from disk.
        
        Args:
            model_dir: Directory containing saved model files
            corpus: Track corpus dictionary
            
        Returns:
            WMFModel instance
        """
        model_dir = Path(model_dir)
        print(f"\nLoading model from {model_dir}/")
        
        # Create instance with minimal initialization
        instance = cls.__new__(cls)
        instance.corpus = corpus
        
        # Load metadata
        with open(model_dir / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        instance.track_ids = metadata['track_ids']
        instance.track_id_to_idx = metadata['track_id_to_idx']
        instance.vocab = metadata['vocab']
        instance.word_to_idx = metadata['word_to_idx']
        instance.num_tracks = metadata['num_tracks']
        
        # Load sparse matrices
        instance.track_tf_matrix = load_npz(model_dir / 'track_tf_matrix.npz')
        
        playlist_matrix_path = model_dir / 'playlist_tf_matrix.npz'
        if playlist_matrix_path.exists():
            playlist_tf_matrix = load_npz(playlist_matrix_path)
            # Convert back to list of rows
            instance.playlist_tf_rows = [playlist_tf_matrix.getrow(i) for i in range(playlist_tf_matrix.shape[0])]
        else:
            instance.playlist_tf_rows = []
        
        # Load ALS model
        als_model_path = model_dir / 'als_model.pkl'
        if als_model_path.exists():
            with open(als_model_path, 'rb') as f:
                instance.model = pickle.load(f)
        else:
            instance.model = None
        
        print("✓ Model loaded successfully")
        return instance


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
    
    # Define model path
    model_path = Config.MODEL_DIR / "wmf"
    model_exists = (model_path / "metadata.pkl").exists()
    
    # Determine whether to train or load
    should_train = Config.RETRAIN or not model_exists
    
    if should_train:
        print(f"\n{'='*70}")
        print("TRAINING NEW MODEL")
        print('='*70)
        
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
            use_gpu=Config.USE_GPU
        )
        
        # Save model
        model.save(model_path)
    else:
        print(f"\n{'='*70}")
        print("LOADING EXISTING MODEL")
        print('='*70)
        
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
        description="Weighted Matrix Factorization Baseline for Music Track Retrieval",
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
        "--model_dir",
        help="Path to directory for saving/loading models"
    )
    parser.add_argument(
        "--n_factors",
        type=int,
        help="Number of latent factors for WMF"
    )
    parser.add_argument(
        "--regularization",
        type=float,
        help="L2 regularization parameter"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Number of ALS iterations"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Confidence weight multiplier (C = 1 + alpha * r)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        help="Number of top results to return per query"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining even if saved model exists"
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    
    args = parser.parse_args()
    Config.update_from_args(args)
    
    main()
