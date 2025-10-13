import json
import math
import os
import numpy as np
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import nltk

from util import load_corpus, load_queries, get_query_files, extract_id, save_results

# Ensure necessary tokenizer resources
nltk.download("punkt", quiet=True)


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Centralized configuration for paths and parameters."""

    # Default paths
    WORKSPACE_DIR = Path("dataset")
    CORPUS_FILE = WORKSPACE_DIR / "tracks_index.json"
    TRAIN_QUERIES_DIR = WORKSPACE_DIR / "train"
    TEST_QUERIES_DIR = WORKSPACE_DIR / "test"
    RESULTS_DIR = WORKSPACE_DIR / "results" / "mf_baseline"

    # Model parameters

    @classmethod
    def initialize(cls):
        """Ensure results directory exists."""
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def update_from_args(cls, args):
        """Update configuration from command line arguments."""
        if args.corpus:
            cls.CORPUS_FILE = Path(args.corpus)
        if args.train_queries:
            cls.TRAIN_QUERIES_DIR = Path(args.train_queries)
        if args.test_queries:
            cls.TEST_QUERIES_DIR = Path(args.test_queries)
        if args.results:
            cls.RESULTS_DIR = Path(args.results)

        # Ensure results directory exists
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def tokenize(text):
    """Simple word tokenizer and normalizer."""
    return [w.lower() for w in nltk.word_tokenize(text) if w.isalnum()]


class MFModel:
    def __init__(self, tracks_corpus):
        self.playlist_names = []  # list of Counter(name tokens)
        self.tracks_tf = {}  # track_id -> Counter(name tokens)
        self.playlist_ids = set()
        self.idx_to_id = {}  # row index -> id
        self.id_to_idx = {}
        self.playlist_cutoff = len(
            tracks_corpus
        )  # cutoff index between tracks and playlists

        # initialize empty Counters for each track
        for idx, track_id in enumerate(tracks_corpus.keys()):
            self.tracks_tf[track_id] = Counter()
            self.idx_to_id[idx] = track_id
            self.id_to_idx[track_id] = idx

        # placeholders for model
        self.vocab_vectorizer = None
        self.U = None
        self.V = None
        self.svd = None

    def add_playlist(self, playlist):
        playlist_id = playlist["pid"]
        self.playlist_ids.add(playlist_id)
        playlist_name = Counter(tokenize(playlist["name"]))
        self.playlist_names.append(playlist_name)

        tracks = playlist.get("tracks", [])
        for track in tracks:
            track_id = extract_id(track["track_uri"])
            if track_id in self.tracks_tf:
                self.tracks_tf[track_id].update(playlist_name)

    def compute_factorization(self, n_components=200):
        """Build TF matrix (songs + playlists) and run truncated SVD."""
        print("Building term-frequency matrix...")

        # 1️⃣ Combine all Counters (songs first, then playlists)
        all_entities = list(self.tracks_tf.values()) + list(self.playlist_names)

        # 2️⃣ Vectorize into TF matrix (rows = songs+playlists, cols = name words)
        self.vocab_vectorizer = DictVectorizer(sparse=True)
        R_name = self.vocab_vectorizer.fit_transform(all_entities)
        print(f"R_name shape: {R_name.shape}")

        # 3️⃣ Factorize
        print("Running truncated SVD...")
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        U = self.svd.fit_transform(R_name)
        V = self.svd.components_.T

        # store embeddings
        self.U = normalize(U)  # rows: entity embeddings (songs+playlists)
        self.V = normalize(V)  # columns: word embeddings
        print("Factorization complete.")

    def infer_from_name(self, playlist_name, top_k=20):
        """Return top_k track IDs relevant to a new playlist name."""
        if self.U is None or self.V is None:
            raise ValueError(
                "Model not yet factorized. Call compute_factorization() first."
            )

        # 1️⃣ Convert name into a word→count dict and vectorize it
        name_tf = Counter(tokenize(playlist_name))
        name_vec = self.vocab_vectorizer.transform([name_tf])  # 1 x vocab_size sparse

        # 2️⃣ Project name into latent space
        name_emb = name_vec.dot(self.V)
        name_emb = normalize(name_emb)

        # 3️⃣ Compute similarity with only SONG rows (exclude playlist part)
        song_embeddings = self.U[: self.playlist_cutoff]
        sims = song_embeddings.dot(name_emb.T).ravel()

        # 4️⃣ Get top_k songs
        top_indices = np.argpartition(-sims, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-sims[top_indices])]
        top_ids = [self.idx_to_id[i] for i in top_indices]

        return top_ids


def main():
    corpus = load_corpus(Config.CORPUS_FILE)
    retriever = MFModel(corpus)

    train_query_files = get_query_files(Config.TRAIN_QUERIES_DIR)
    print(f"Found {len(train_query_files)} training query files.")
    for query_file in train_query_files:
        print(f"\nProcessing training queries from: {query_file.name}")
        queries = load_queries(query_file)
        for query in tqdm(queries, desc="Adding playlists"):
            retriever.add_playlist(query)
    
    retriever.compute_factorization()

    test_query_files = get_query_files(Config.TEST_QUERIES_DIR)
    print(f"\nFound {len(test_query_files)} test query files.")
    for query_file in test_query_files:
        print(f"\nProcessing test queries from: {query_file.name}")
        queries = load_queries(query_file)
        results = {}
        for query in tqdm(queries, desc="Retrieving"):
            playlist_name = query.get("name", "")
            pid = query.get("pid")
            top_k_ids = retriever.infer_from_name(playlist_name, top_k=100)
            top_k_tracks = [corpus[tid] for tid in top_k_ids]
            results[pid] = top_k_tracks

        save_results(results, query_file.name, Config.RESULTS_DIR)

if __name__ == "__main__":
    Config.initialize()
    main()
