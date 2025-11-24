#!/usr/bin/env python3
"""
Interactive inference script for playlist recommendation system.
Loads models once and allows continuous querying until user exits (Ctrl+C).
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict, Counter
import json
import pickle
import signal

# For dense retrieval
from sentence_transformers import SentenceTransformer
import faiss

from util import load_corpus, tokenize


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Centralized configuration for paths and parameters."""
    
    WORKSPACE_DIR = Path("dataset")
    CORPUS_FILE = WORKSPACE_DIR / "track_corpus.json"
    METADATA_FILE = WORKSPACE_DIR / "playlist_metadata.json"
    INVERTED_INDEX = WORKSPACE_DIR / "inverted_index.json"
    # CHANGE THIS TO YOUR SAVE SPACE FOR HYBRID MODEL
    MODEL_DIR = Path("/scratch4/workspace/oyilmazel_umass_edu-mpd/models/hybrid/")
    
    DENSE_MODEL = "all-MiniLM-L6-v2"
    TOP_K = 20  # Default number of recommendations
    CANDIDATE_K = 1000
    
    # Default weights
    DENSE_WEIGHT = 1.0
    LMIR_WEIGHT = 0.0
    COOCCUR_WEIGHT = 0.0
    GF_WEIGHT = 0.0
    
    MU = 2000

    @classmethod
    def update_from_args(cls, args):
        """Update configuration from command line arguments."""
        if args.workspace:
            cls.WORKSPACE_DIR = Path(args.workspace)
            cls.CORPUS_FILE = cls.WORKSPACE_DIR / "track_corpus.json"
            cls.METADATA_FILE = cls.WORKSPACE_DIR / "playlist_metadata.json"
            cls.INVERTED_INDEX = cls.WORKSPACE_DIR / "inverted_index.json"
        if args.model_dir:
            cls.MODEL_DIR = Path(args.model_dir)
        if args.top_k:
            cls.TOP_K = args.top_k


# ============================================================================
# LIGHTWEIGHT RETRIEVERS (for inference only)
# ============================================================================
class DenseRetriever:
    """Dense retrieval using sentence transformers and FAISS."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.track_ids = []
    
    def retrieve(self, query_text, top_k=100):
        """Retrieve top-k tracks using dense similarity."""
        query_embedding = self.model.encode(
            [query_text], convert_to_numpy=True, normalize_embeddings=True
        )
        
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.track_ids):
                results.append((self.track_ids[idx], float(score)))
        
        return results
    
    def load(self, save_dir):
        """Load index and metadata."""
        save_dir = Path(save_dir)
        self.index = faiss.read_index(str(save_dir / "faiss_index.bin"))
        with open(save_dir / "track_ids.pkl", "rb") as f:
            self.track_ids = pickle.load(f)
        print(f"Dense retriever loaded from {save_dir}")


class DirichletLMRetriever:
    """Dirichlet Language Model retriever using inverted index."""
    
    def __init__(self, mu=2000):
        self.mu = mu
        self.inverted_index = {}
        self.doc_lens = {}
        self.collection_freq = Counter()
        self.collection_len = 0
        self.total_docs = 0
    
    def load_inverted_index(self, index_path):
        """Load inverted index from JSON file."""
        print(f"Loading inverted index from {index_path}...")
        with open(index_path, "r") as f:
            data = json.load(f)
        
        self.inverted_index = data
        self.doc_lens = data["DOC_LENGTHS"]
        self.total_docs = data["TOTAL_DOCS"]
        
        for term, postings in self.inverted_index.items():
            if term in ["DOCID_MAP", "DOC_LENGTHS", "TOTAL_DOCS"]:
                continue
            cf = sum(tf for _, tf in postings)
            self.collection_freq[term] = cf
            self.collection_len += cf
        
        print("Inverted index loaded")
    
    def retrieve(self, query_text, top_k=10):
        """Retrieve and rank documents for a query."""
        import math
        query_tokens = tokenize(query_text)
        scores = defaultdict(float)
        
        P_tc = {
            t: self.collection_freq[t] / self.collection_len
            for t in query_tokens
            if t in self.collection_freq
        }
        
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


class CooccurrenceRetriever:
    """Retriever based on track co-occurrence patterns."""
    
    def __init__(self):
        self.track_cooccurrence = defaultdict(Counter)
        self.track_popularity = Counter()
    
    def retrieve(self, query_text, top_k=100, seed_tracks=None):
        """Retrieve tracks based on co-occurrence with seed tracks."""
        if seed_tracks is None or len(seed_tracks) == 0:
            ranked = self.track_popularity.most_common(top_k)
            return [(track_id, float(score)) for track_id, score in ranked]
        
        scores = Counter()
        for seed_track in seed_tracks:
            if seed_track in self.track_cooccurrence:
                for track_id, count in self.track_cooccurrence[seed_track].items():
                    scores[track_id] += count
        
        ranked = scores.most_common(top_k)
        return [(track_id, float(score)) for track_id, score in ranked]
    
    def load(self, save_dir):
        """Load co-occurrence data."""
        save_dir = Path(save_dir)
        with open(save_dir / "cooccurrence.pkl", "rb") as f:
            data = pickle.load(f)
        
        self.track_cooccurrence = defaultdict(
            Counter, {k: Counter(v) for k, v in data["track_cooccurrence"].items()}
        )
        self.track_popularity = Counter(data["track_popularity"])
        print(f"✓ Co-occurrence retriever loaded from {save_dir}")


class GenreFeatureRetriever:
    """Retriever based on genre and audio feature similarity."""
    
    def __init__(self):
        self.track_features = {}
        self.feature_to_tracks = defaultdict(set)
        self._feature_idf = {}
        self._total_tracks = 0
    
    def retrieve(self, query_text, top_k=100, features=[]):
        """Retrieve based on genre and feature similarity."""
        import math
        if not features:
            return []
        
        tokenized_features = []
        for ft in features:
            tokenized_features.extend(tokenize(ft))
        
        if not tokenized_features:
            return []
        
        scores = Counter()
        for token_ft in tokenized_features:
            if token_ft in self.feature_to_tracks:
                idf = self._feature_idf.get(token_ft, 1.0)
                for track_id in self.feature_to_tracks[token_ft]:
                    scores[track_id] += idf
        
        num_query_features = len(tokenized_features)
        results = []
        for track_id, score in scores.most_common(top_k * 2):
            track_feature_count = len(self.track_features.get(track_id, set()))
            if track_feature_count > 0:
                normalized_score = score / (
                    num_query_features * math.log2(1 + 1 / track_feature_count)
                )
                results.append((track_id, normalized_score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def load(self, save_dir):
        """Load genre/feature data."""
        save_dir = Path(save_dir)
        with open(save_dir / "genre_features.pkl", "rb") as f:
            data = pickle.load(f)
        
        self.track_features = data["track_features"]
        self.feature_to_tracks = defaultdict(
            set, {k: set(v) for k, v in data["feature_to_tracks"].items()}
        )
        self._feature_idf = data.get("feature_idf", {})
        self._total_tracks = data.get("total_tracks", len(self.track_features))
        print(f"✓ Genre/feature retriever loaded from {save_dir}")


# ============================================================================
# HYBRID RETRIEVER (inference only)
# ============================================================================
class HybridRetriever:
    """Combines multiple retrieval methods for inference."""
    
    def __init__(self, corpus, metadata):
        self.corpus = corpus
        self.metadata = metadata
        self.dense_retriever = DenseRetriever(Config.DENSE_MODEL)
        self.lmir_retriever = DirichletLMRetriever(mu=Config.MU)
        self.cooccur_retriever = CooccurrenceRetriever()
        self.gf_retriever = GenreFeatureRetriever()
        
        # Weights (loaded from saved model)
        self.dense_weight = Config.DENSE_WEIGHT
        self.lmir_weight = Config.LMIR_WEIGHT
        self.cooccur_weight = Config.COOCCUR_WEIGHT
        self.gf_weight = Config.GF_WEIGHT
    
    def load(self, save_dir):
        """Load all retrievers."""
        save_dir = Path(save_dir)
        
        print("\nLoading hybrid retrieval model...")
        
        # Load weights
        with open(save_dir / "weights.json", "r") as f:
            weights = json.load(f)
            self.dense_weight = weights["dense_weight"]
            self.lmir_weight = weights["lmir_weight"]
            self.cooccur_weight = weights["cooccur_weight"]
            self.gf_weight = weights["gf_weight"]
        
        print(f"Model weights: Dense={self.dense_weight}, LMIR={self.lmir_weight}, "
              f"Cooccur={self.cooccur_weight}, GenreFeature={self.gf_weight}")
        
        # Load retrievers
        self.dense_retriever.load(save_dir / "dense")
        
        if Config.INVERTED_INDEX.exists():
            self.lmir_retriever.load_inverted_index(Config.INVERTED_INDEX)
        else:
            print(f"Warning: LMIR index not found at {Config.INVERTED_INDEX}")
        
        self.cooccur_retriever.load(save_dir / "cooccur")
        self.gf_retriever.load(save_dir / "gf")
        
        print("✓ All models loaded successfully!\n")
    
    def retrieve(self, query_text, features=None, top_k=20):
        """Hybrid retrieval with score fusion."""
        candidate_k = Config.CANDIDATE_K
        
        if features is None:
            features = []
        
        # Get candidates from each retriever
        dense_results = self.dense_retriever.retrieve(query_text, top_k=candidate_k)
        lmir_results = self.lmir_retriever.retrieve(query_text, top_k=candidate_k)
        gf_results = self.gf_retriever.retrieve(
            query_text, top_k=candidate_k, features=features
        )
        
        # Use top LMIR results as seeds for co-occurrence
        seed_tracks = [track_id for track_id, _ in lmir_results[:50]]
        cooccur_results = self.cooccur_retriever.retrieve(
            query_text, top_k=candidate_k, seed_tracks=seed_tracks
        )
        
        # Fuse scores
        fused_scores = self._fuse_scores(
            dense_results, lmir_results, cooccur_results, gf_results
        )
        
        # Sort and return top-k
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
    
    def _fuse_scores(self, dense_results, lmir_results, cooccur_results, gf_results):
        """Normalize and fuse scores from multiple retrievers."""
        
        def normalize_scores(results):
            if not results:
                return {}
            scores = {track_id: score for track_id, score in results}
            min_score = min(scores.values())
            max_score = max(scores.values())
            
            if max_score == min_score:
                return {track_id: 1.0 for track_id in scores}
            
            return {
                track_id: (score - min_score) / (max_score - min_score)
                for track_id, score in scores.items()
            }
        
        dense_norm = normalize_scores(dense_results)
        lmir_norm = normalize_scores(lmir_results)
        cooccur_norm = normalize_scores(cooccur_results)
        gf_norm = normalize_scores(gf_results)
        
        all_tracks = (
            set(dense_norm.keys())
            | set(lmir_norm.keys())
            | set(cooccur_norm.keys())
            | set(gf_norm.keys())
        )
        
        fused = {}
        for track_id in all_tracks:
            score = (
                self.dense_weight * dense_norm.get(track_id, 0.0)
                + self.lmir_weight * lmir_norm.get(track_id, 0.0)
                + self.cooccur_weight * cooccur_norm.get(track_id, 0.0)
                + self.gf_weight * gf_norm.get(track_id, 0.0)
            )
            fused[track_id] = score
        
        return fused


# ============================================================================
# INTERACTIVE SESSION
# ============================================================================
class InteractiveSession:
    """Interactive retrieval session."""
    
    def __init__(self, retriever, corpus, metadata):
        self.retriever = retriever
        self.corpus = corpus
        self.metadata = metadata
        self.running = True
        
        # Setup signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\n👋 Goodbye! Thanks for using the playlist recommender.")
        self.running = False
        sys.exit(0)
    
    def format_track(self, track_id, rank, score):
        """Format track information for display."""
        track = self.corpus.get(track_id, {})
        track_name = track.get("track_name", "Unknown")
        artist_name = track.get("artist_name", "Unknown")
        album_name = track.get("album_name", "Unknown")
        
        return f"  {rank:2d}. {track_name} - {artist_name} [{album_name}] (score: {score:.3f})"
    
    def run(self):
        """Run interactive session."""
        print("=" * 70)
        print("🎵 INTERACTIVE PLAYLIST RECOMMENDATION SYSTEM 🎵")
        print("=" * 70)
        print("\nEnter a playlist description and get song recommendations!")
        print("Commands:")
        print("  - Type your playlist description and press Enter")
        print("  - Type 'quit' or 'exit' to end session")
        print("  - Press Ctrl+C to exit anytime")
        print("=" * 70)
        
        while self.running:
            try:
                # Get user input
                print("\n" + "─" * 70)
                query = input("🎧 Enter playlist description: ").strip()
                
                if not query:
                    print("⚠️  Please enter a description!")
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye! Thanks for using the playlist recommender.")
                    break
                
                # Optional: Ask for features/genres
                features_input = input("🎸 Enter genres/features (optional, comma-separated): ").strip()
                features = []
                if features_input:
                    features = [f.strip() for f in features_input.split(',') if f.strip()]
                
                # Retrieve recommendations
                print("\n🔍 Finding recommendations...")
                results = self.retriever.retrieve(
                    query, 
                    features=features, 
                    top_k=Config.TOP_K
                )
                
                if not results:
                    print("❌ No recommendations found. Try a different description!")
                    continue
                
                # Display results
                print(f"\n✨ Top {len(results)} Recommendations:\n")
                for rank, (track_id, score) in enumerate(results, 1):
                    print(self.format_track(track_id, rank, score))
                
            except KeyboardInterrupt:
                self.signal_handler(None, None)
            except EOFError:
                print("\n\n👋 Goodbye! Thanks for using the playlist recommender.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again with a different query.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Interactive playlist recommendation system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace", help="Path to workspace directory")
    parser.add_argument("--model_dir", help="Path to model directory")
    parser.add_argument("--top_k", type=int, help="Number of recommendations to show")
    
    args = parser.parse_args()
    Config.update_from_args(args)
    
    # Check if model exists
    if not Config.MODEL_DIR.exists():
        print(f"❌ Error: Model directory not found at {Config.MODEL_DIR}")
        print("Please train the model first by running the main training script.")
        sys.exit(1)
    
    # Load corpus and metadata
    print("Loading corpus and metadata...")
    corpus = load_corpus(Config.CORPUS_FILE)
    metadata = load_corpus(Config.METADATA_FILE) if Config.METADATA_FILE.exists() else {}
    print(f"Loaded {len(corpus)} tracks")
    
    # Initialize and load retriever
    retriever = HybridRetriever(corpus, metadata)
    retriever.load(Config.MODEL_DIR)
    
    # Start interactive session
    session = InteractiveSession(retriever, corpus, metadata)
    session.run()


if __name__ == "__main__":
    main()
