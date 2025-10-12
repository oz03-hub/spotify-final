import json
import math
import os
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import nltk

from util import load_corpus, load_queries, get_query_files, extract_id

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
        self.playlist_names = []
        self.tracks_tf = {}

        self.idx_to_id = {}
        self.playlist_cutoff = len(tracks_corpus) # indexes after this are playlists

        for track in tracks_corpus:
            track_id = extract_id(track["track_uri"])
            self.tracks_tf[track_id] = Counter()

    def add_playlist(self, playlist):
        playlist_id = playlist["id"]
        self.playlist_ids.add(playlist_id)

        playlist_name = Counter(tokenize(playlist["name"]))
        self.playlist_names.append(playlist_name)
        tracks = playlist.get("tracks", [])
        for track in tracks:
            track_id = extract_id(track["track_uri"])
            if track_id in self.tracks_tf:
                self.tracks_tf[track_id].update(playlist_name)

    def compute_factorization(self):
        pass

def main():
    corpus = load_corpus(Config.CORPUS_FILE)
    retriever = MFModel(corpus)

    train_query_files = get_query_files(Config.TRAIN_QUERIES_DIR)
    print(f"Found {len(train_query_files)} training query files.")
    for query_file in train_query_files:
        print(f"\nProcessing training queries from: {query_file.name}")
        queries = load_queries(query_file)
        for query in tqdm(queries["playlists"], desc="Adding playlists"):
            retriever.add_playlist(query)
    
