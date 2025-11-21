import argparse
import numpy as np
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
import json
import math

# For dense retrieval
from sentence_transformers import SentenceTransformer
import faiss

from util import load_corpus, load_queries, get_query_files, save_results, tokenize


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Centralized configuration for paths and parameters."""

    # Default paths
    WORKSPACE_DIR = Path("dataset")
    CORPUS_FILE = WORKSPACE_DIR / "track_corpus.json"
    TRAIN_QUERIES_DIR = WORKSPACE_DIR / "train"
    INVERTED_INDEX = WORKSPACE_DIR / "inverted_index.json"
    RESULTS_DIR = WORKSPACE_DIR / "results" / "hybrid_baseline_zeroshot"
    MODEL_DIR = Path("model")

    METADATA = WORKSPACE_DIR / "playlist_metadata.json"

    # Model parameters
    # COMMENT OUT IF YOU WANT 0-SHOT
    DENSE_MODEL = "all-MiniLM-L6-v2"  # Fast and effective sentence transformer
    # DENSE_MODEL = "/scratch4/workspace/oyilmazel_umass_edu-mpd/transformer_finetuned/best" # CHANGE THIS TO YOUR FINE-TUNED LOCATION
    TOP_K = 100
    CANDIDATE_K = 1000  # Retrieve more candidates before re-ranking

    # Fusion weights
    DENSE_WEIGHT = 1.0
    LMIR_WEIGHT = 0.0
    COOCCUR_WEIGHT = 0.0
    GF_WEIGHT = 0.0

    MU = 2000

    # Training options
    RETRAIN = False

    @classmethod
    def update_from_args(cls, args):
        """Update configuration from command line arguments."""
        if args.workspace:
            cls.WORKSPACE_DIR = Path(args.workspace)
            cls.CORPUS_FILE = cls.WORKSPACE_DIR / "track_corpus.json"
            cls.TRAIN_QUERIES_DIR = cls.WORKSPACE_DIR / "train"
            cls.INVERTED_INDEX = cls.WORKSPACE_DIR / "inverted_index.json"
        if args.corpus:
            cls.CORPUS_FILE = Path(args.corpus)
        if args.train_queries:
            cls.TRAIN_QUERIES_DIR = Path(args.train_queries)
        if args.results:
            cls.RESULTS_DIR = Path(args.results)
        if args.model_dir:
            cls.MODEL_DIR = Path(args.model_dir)
        if args.inverted_index:
            cls.INVERTED_INDEX = Path(args.inverted_index)
        if args.top_k:
            cls.TOP_K = args.top_k
        if args.retrain:
            cls.RETRAIN = True

        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)


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


# ============================================================================
# DENSE RETRIEVER (Sentence Transformers + FAISS)
# ============================================================================
class DenseRetriever:
    """Dense retrieval using sentence transformers and FAISS."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.track_ids = []

    def build_index(self, corpus):
        """Build FAISS index from track corpus."""
        print("Building dense embeddings for tracks...")
        self.track_ids = list(corpus.keys())

        # Create text representations for tracks
        texts = []
        for track_id in tqdm(self.track_ids, desc="Preparing texts"):
            track = corpus[track_id]
            text = f"{track.get('track_name')} {track.get('artist_name')} {track.get('album_name')} {track.get('extended_text', '')}"
            texts.append(text)

        # Encode in batches
        print("Encoding tracks...")
        embeddings = self.model.encode(
            texts,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )

        print("Building FAISS IVF index...")
        n_tracks = len(self.track_ids)

        nlist = min(int(4 * np.sqrt(n_tracks)), n_tracks // 10)
        nlist = max(nlist, 1)

        # Quantizer for IVF
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        self.index = faiss.IndexIVFFlat(
            quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT
        )

        # Train the index
        print(f"Training IVF index with {nlist} clusters...")
        self.index.train(embeddings.astype("float32"))
        self.index.add(embeddings.astype("float32"))

        # Set nprobe for search (trade-off between speed and accuracy)
        self.index.nprobe = min(nlist // 4, 32)  # Search 25% of clusters, max 32

        print(
            f"Index built with {len(self.track_ids)} tracks, nlist={nlist}, nprobe={self.index.nprobe}"
        )

    def retrieve(self, query_text, top_k=100):
        """Retrieve top-k tracks using dense similarity."""
        query_embedding = self.model.encode(
            [query_text], convert_to_numpy=True, normalize_embeddings=True
        )

        scores, indices = self.index.search(query_embedding.astype("float32"), top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.track_ids):  # Valid index
                results.append((self.track_ids[idx], float(score)))

        return results

    def retrieve_batch(self, query_texts, top_k=100):
        """Retrieve top-k tracks for multiple queries at once."""
        # Encode all queries in one batch
        query_embeddings = self.model.encode(
            query_texts,
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(query_texts) > 100,
        )

        # Search all queries at once
        scores, indices = self.index.search(query_embeddings.astype("float32"), top_k)

        # Convert to results format
        all_results = []
        for i in range(len(query_texts)):
            results = []
            for idx, score in zip(indices[i], scores[i]):
                if idx >= 0 and idx < len(
                    self.track_ids
                ):  # Valid index (-1 means not found)
                    results.append((self.track_ids[idx], float(score)))
            all_results.append(results)

        return all_results

    def save(self, save_dir):
        """Save index and metadata."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(save_dir / "faiss_index.bin"))

        # Save track IDs
        with open(save_dir / "track_ids.pkl", "wb") as f:
            pickle.dump(self.track_ids, f)

        print(f"Dense retriever saved to {save_dir}")

    def load(self, save_dir):
        """Load index and metadata."""
        save_dir = Path(save_dir)

        # Load FAISS index
        self.index = faiss.read_index(str(save_dir / "faiss_index.bin"))

        # Load track IDs
        with open(save_dir / "track_ids.pkl", "rb") as f:
            self.track_ids = pickle.load(f)

        print(f"Dense retriever loaded from {save_dir}")


# ============================================================================
# PLAYLIST CO-OCCURRENCE RETRIEVER
# ============================================================================
class CooccurrenceRetriever:
    """Retriever based on track co-occurrence patterns in training playlists."""

    def __init__(self):
        self.track_cooccurrence = defaultdict(
            Counter
        )  # track_id -> {co_track_id: count}
        self.playlist_tracks = {}  # pid -> [track_ids]
        self.track_to_playlists = defaultdict(list)  # track_id -> [pids]
        self.track_popularity = Counter()

    def build_from_playlists(self, training_playlists):
        """Build co-occurrence matrix from training playlists."""
        print("Building co-occurrence patterns from training playlists...")

        for playlist in tqdm(training_playlists, desc="Processing playlists"):
            pid = playlist.get("pid")
            track_ids = [
                t["track_uri"].split(":")[2] for t in playlist.get("tracks", [])
            ]

            self.playlist_tracks[pid] = track_ids

            # Track popularity
            for track_id in track_ids:
                self.track_popularity[track_id] += 1
                self.track_to_playlists[track_id].append(pid)

            # Co-occurrence: each pair of tracks in playlist
            for i, track_id_1 in enumerate(track_ids):
                for track_id_2 in track_ids[i + 1 :]:
                    self.track_cooccurrence[track_id_1][track_id_2] += 1
                    self.track_cooccurrence[track_id_2][track_id_1] += 1

        print(f"Built co-occurrence matrix for {len(self.track_cooccurrence)} tracks")
        print(f"Total playlists: {len(self.playlist_tracks)}")

    def retrieve(self, query_text, top_k=100, seed_tracks=None):
        """
        Retrieve tracks based on co-occurrence with seed tracks.

        If seed_tracks is None, we use a simple approach: find tracks mentioned
        in playlists with similar names (lexical match).
        """
        if seed_tracks is None or len(seed_tracks) == 0:
            # Fallback: find playlists with overlapping words in query
            # query_tokens = set(tokenize(query_text))
            candidate_tracks = Counter()

            # This is a simple heuristic - you could also use the dense retriever
            # to find similar playlists first
            for track_id in self.track_popularity.keys():
                candidate_tracks[track_id] = self.track_popularity[track_id]

            ranked = candidate_tracks.most_common(top_k)
            return [(track_id, float(score)) for track_id, score in ranked]

        # If we have seed tracks, use co-occurrence
        scores = Counter()
        for seed_track in seed_tracks:
            if seed_track in self.track_cooccurrence:
                for track_id, count in self.track_cooccurrence[seed_track].items():
                    scores[track_id] += count

        ranked = scores.most_common(top_k)
        return [(track_id, float(score)) for track_id, score in ranked]

    def save(self, save_dir):
        """Save co-occurrence data."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "cooccurrence.pkl", "wb") as f:
            pickle.dump(
                {
                    "track_cooccurrence": dict(self.track_cooccurrence),
                    "playlist_tracks": self.playlist_tracks,
                    "track_to_playlists": dict(self.track_to_playlists),
                    "track_popularity": dict(self.track_popularity),
                },
                f,
            )

        print(f"Co-occurrence retriever saved to {save_dir}")

    def load(self, save_dir):
        """Load co-occurrence data."""
        save_dir = Path(save_dir)

        with open(save_dir / "cooccurrence.pkl", "rb") as f:
            data = pickle.load(f)

        self.track_cooccurrence = defaultdict(
            Counter, {k: Counter(v) for k, v in data["track_cooccurrence"].items()}
        )
        self.playlist_tracks = data["playlist_tracks"]
        self.track_to_playlists = defaultdict(list, data["track_to_playlists"])
        self.track_popularity = Counter(data["track_popularity"])

        print(f"Co-occurrence retriever loaded from {save_dir}")


# ============================================================================
# GENRE/FEATURE RETRIEVER (OPTIMIZED)
# ============================================================================
class GenreFeatureRetriever:
    """Optimized retriever based on genre and audio feature similarity."""

    def __init__(self):
        self.track_features = {}  # track_id -> set of features
        self.feature_to_tracks = defaultdict(set)

        # Pre-computed IDF values
        self._feature_idf = {}
        self._total_tracks = 0

    def build_from_corpus(self, corpus):
        """Build genre/feature associations from corpus."""
        print("Building genre/feature associations from playlist metadata...")

        for track_id, track_obj in tqdm(corpus.items(), desc="Processing corpus"):
            all_features = track_obj.get("features", [])

            if all_features:
                feature_set = set(all_features)
                self.track_features[track_id] = feature_set

                for ft in feature_set:
                    self.feature_to_tracks[ft].add(track_id)

        self._total_tracks = len(self.track_features)

        # Pre-compute IDF values
        for feature, tracks in self.feature_to_tracks.items():
            df = len(tracks)
            self._feature_idf[feature] = math.log(self._total_tracks / (1 + df))

        print(f"Built associations for {len(self.track_features)} tracks")
        print(f"Built associations for {len(self.feature_to_tracks)} features")

    def retrieve(self, query_text, top_k=100, features=[]):
        """Optimized retrieval based on genre and feature similarity."""
        if not features:
            return []

        # Tokenize features
        tokenized_features = []
        for ft in features:
            tokenized_features.extend(tokenize(ft))

        if not tokenized_features:
            return []

        # Use Counter for efficient accumulation
        scores = Counter()

        for token_ft in tokenized_features:
            if token_ft in self.feature_to_tracks:
                idf = self._feature_idf.get(token_ft, 1.0)
                for track_id in self.feature_to_tracks[token_ft]:
                    scores[track_id] += idf

        # Normalize by number of query features and track feature count
        num_query_features = len(tokenized_features)
        results = []
        for track_id, score in scores.most_common(top_k * 2):  # Get extra for filtering
            track_feature_count = len(self.track_features.get(track_id, set()))
            if track_feature_count > 0:
                normalized_score = score / (
                    num_query_features * math.log2(1 + 1 / track_feature_count)
                )
                results.append((track_id, normalized_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save(self, save_dir):
        """Save genre/feature data."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "genre_features.pkl", "wb") as f:
            pickle.dump(
                {
                    "track_features": self.track_features,
                    "feature_to_tracks": {
                        k: list(v) for k, v in self.feature_to_tracks.items()
                    },
                    "feature_idf": self._feature_idf,
                    "total_tracks": self._total_tracks,
                },
                f,
            )

        print(f"Genre/feature retriever saved to {save_dir}")

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

        print(f"Genre/feature retriever loaded from {save_dir}")


# ============================================================================
# HYBRID RETRIEVER (combines all methods)
# ============================================================================
class HybridRetriever:
    """Combines dense, sparse (LMIR), and co-occurrence retrievers."""

    def __init__(
        self,
        corpus,
        metadata,
        dense_weight=0.25,
        lmir_weight=0.5,
        cooccur_weight=0.25,
        gf_weight=0.0,
    ):
        self.corpus = corpus
        self.dense_weight = dense_weight
        self.lmir_weight = lmir_weight
        self.cooccur_weight = cooccur_weight
        self.gf_weight = gf_weight

        self.metadata = metadata

        self.dense_retriever = DenseRetriever(Config.DENSE_MODEL)
        self.lmir_retriever = DirichletLMRetriever(mu=Config.MU)
        self.cooccur_retriever = CooccurrenceRetriever()
        self.gf_retriever = GenreFeatureRetriever()

    def build(self, training_playlists):
        """Build all retrievers."""
        self.gf_retriever.build_from_corpus(self.corpus)

        # Build dense index
        self.dense_retriever.build_index(self.corpus)

        if Config.INVERTED_INDEX.exists():
            self.lmir_retriever.load_inverted_index(Config.INVERTED_INDEX)
        else:
            print(f"Warning: LMIR index not found at {Config.INVERTED_INDEX}")

        # Build co-occurrence
        self.cooccur_retriever.build_from_playlists(training_playlists)

    def retrieve(self, query_text, features, top_k=100):
        """
        Hybrid retrieval with score fusion.

        Strategy:
        1. Get top-K candidates from each retriever
        2. Normalize scores to [0, 1]
        3. Weighted fusion
        4. Return top-K
        """
        candidate_k = Config.CANDIDATE_K

        # Get candidates from each retriever
        dense_results = self.dense_retriever.retrieve(query_text, top_k=candidate_k)
        lmir_results = self.lmir_retriever.retrieve(query_text, top_k=candidate_k)

        gf_results = self.gf_weight.retrieve(
            query_text, top_k=candidate_k, features=features
        )

        # For co-occurrence, we can use top dense results as seeds
        seed_tracks = [track_id for track_id, _ in lmir_results[:50]]
        cooccur_results = self.cooccur_retriever.retrieve(
            query_text, top_k=candidate_k, seed_tracks=seed_tracks
        )

        # Normalize scores and fuse
        fused_scores = self._fuse_scores(
            dense_results, lmir_results, cooccur_results, gf_results
        )

        # Sort by fused score
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def retrieve_batch(self, query_texts, pids, top_k=100):
        """
        Batch hybrid retrieval for multiple queries.
        """
        candidate_k = Config.CANDIDATE_K
        n_queries = len(query_texts)

        # Batch dense retrieval (biggest speedup)
        print(f"Dense retrieval for {n_queries} queries...")
        all_dense_results = self.dense_retriever.retrieve_batch(
            query_texts, top_k=candidate_k
        )

        # LMIR and co-occurrence still need per-query processing
        # but we can parallelize or optimize them separately
        all_results = []

        for i, query_text in enumerate(tqdm(query_texts, desc="Hybrid fusion")):
            dense_results = all_dense_results[i]
            meta_obj = self.metadata.get(pids[i], {}).get("label", {})
            features = meta_obj.get("genres", []) + meta_obj.get("features", [])

            gf_results = self.gf_retriever.retrieve(
                query_text, top_k=candidate_k, features=features
            )
            lmir_results = self.lmir_retriever.retrieve(query_text, top_k=candidate_k)

            # Use top LMIR results as seeds for co-occurrence
            seed_tracks = [track_id for track_id, _ in lmir_results[:50]]
            cooccur_results = self.cooccur_retriever.retrieve(
                query_text, top_k=candidate_k, seed_tracks=seed_tracks
            )

            # Fuse scores
            fused_scores = self._fuse_scores(
                dense_results, lmir_results, cooccur_results, gf_results
            )
            ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            all_results.append(ranked[:top_k])

        return all_results

    def _fuse_scores(self, dense_results, lmir_results, cooccur_results, gf_results):
        """Normalize and fuse scores from multiple retrievers."""

        def normalize_scores(results):
            """Min-max normalization to [0, 1]."""
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

        # Normalize each retriever's scores
        dense_norm = normalize_scores(dense_results)
        lmir_norm = normalize_scores(lmir_results)
        cooccur_norm = normalize_scores(cooccur_results)
        gf_norm = normalize_scores(gf_results)

        # Get all candidate tracks
        all_tracks = (
            set(dense_norm.keys())
            | set(lmir_norm.keys())
            | set(cooccur_norm.keys())
            | set(gf_norm.keys())
        )

        # Fuse scores
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

    def save(self, save_dir):
        """Save all retrievers."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.gf_retriever.save(save_dir / "gf")

        self.dense_retriever.save(save_dir / "dense")
        self.cooccur_retriever.save(save_dir / "cooccur")

        # Save weights
        with open(save_dir / "weights.json", "w") as f:
            json.dump(
                {
                    "dense_weight": self.dense_weight,
                    "lmir_weight": self.lmir_weight,
                    "cooccur_weight": self.cooccur_weight,
                    "gf_weight": self.gf_weight,
                },
                f,
            )

        print(f"Hybrid retriever saved to {save_dir}")

    def load(self, save_dir):
        """Load all retrievers."""
        save_dir = Path(save_dir)

        self.dense_retriever.load(save_dir / "dense")

        if Config.INVERTED_INDEX.exists():
            self.lmir_retriever.load_inverted_index(Config.INVERTED_INDEX)

        self.cooccur_retriever.load(save_dir / "cooccur")
        self.gf_retriever.load(save_dir / "gf")

        # Load weights
        with open(save_dir / "weights.json", "r") as f:
            weights = json.load(f)
            self.dense_weight = weights["dense_weight"]
            self.lmir_weight = weights["lmir_weight"]
            self.cooccur_weight = weights["cooccur_weight"]
            self.gf_weight = weights["gf_weight"]

        print(f"Hybrid retriever loaded from {save_dir}")


# ============================================================================
# RESULTS PROCESSING
# ============================================================================
# def process_queries(retriever, queries, corpus, top_k):
#     """Process all queries and return results."""
#     results = {}
#     for playlist in tqdm(queries, desc="Retrieving"):
#         query_text = f"{playlist.get('name', '')} {playlist.get('description', '')}"
#         pid = playlist.get("pid")
#         retrieved = retriever.retrieve(query_text, top_k=top_k)

#         track_results = []
#         for track_id, _ in retrieved:
#             if track_id in corpus:
#                 track = {
#                     "track_uri": corpus[track_id]["track_uri"],
#                     "artist_uri": corpus[track_id]["artist_uri"],
#                 }
#                 track_results.append(track)
#         results[pid] = track_results
#     return results


def process_queries(retriever, queries, corpus, top_k):
    """Process all queries using batch retrieval."""
    # Prepare all query texts
    query_texts = []
    pids = []
    for playlist in queries:
        query_text = f"{playlist.get('name', '')} {playlist.get('description', '')}".strip()
        query_texts.append(query_text)
        pids.append(playlist.get("pid"))

    # Batch retrieval
    all_retrieved = retriever.retrieve_batch(query_texts, pids, top_k=top_k)

    # Format results
    results = {}
    for pid, retrieved in zip(pids, all_retrieved):
        track_results = []
        for track_id, _ in retrieved:
            if track_id in corpus:
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
    print(f"Loading corpus from {Config.CORPUS_FILE}")
    corpus = load_corpus(Config.CORPUS_FILE)

    metadata = load_corpus(Config.METADATA)

    model_path = Config.MODEL_DIR / "hybrid"
    model_exists = (model_path / "weights.json").exists()

    should_train = Config.RETRAIN or not model_exists

    if should_train:
        print(f"\n{'=' * 70}")
        print("TRAINING NEW HYBRID MODEL")
        print("=" * 70)

        # Initialize hybrid retriever
        retriever = HybridRetriever(
            corpus,
            metadata=metadata,
            dense_weight=Config.DENSE_WEIGHT,
            lmir_weight=Config.LMIR_WEIGHT,
            cooccur_weight=Config.COOCCUR_WEIGHT,
            gf_weight=Config.GF_WEIGHT,
        )

        # Load all training playlists
        print("\nLoading training playlists...")
        train_query_files = get_query_files(Config.TRAIN_QUERIES_DIR)
        all_training_playlists = []

        for query_file in train_query_files:
            playlists = load_queries(query_file)
            all_training_playlists.extend(playlists)

        print(f"Loaded {len(all_training_playlists)} training playlists")

        # Build all retrievers
        retriever.build(all_training_playlists)

        # Save model
        retriever.save(model_path)

    else:
        print(f"\n{'=' * 70}")
        print("LOADING EXISTING HYBRID MODEL")
        print("=" * 70)

        retriever = HybridRetriever(
            corpus,
            metadata=metadata,
            dense_weight=Config.DENSE_WEIGHT,
            lmir_weight=Config.LMIR_WEIGHT,
            cooccur_weight=Config.COOCCUR_WEIGHT,
            gf_weight=Config.GF_WEIGHT,
        )
        retriever.load(model_path)

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
            results = process_queries(retriever, queries, corpus, Config.TOP_K)
            save_results(results, query_file.name, split_result_dir)

    print("\nAll queries processed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid Dense-Sparse-Collaborative Retrieval for Cold-Start Playlists",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace", help="Path to workspace directory")
    parser.add_argument("--corpus", help="Path to corpus JSON file")
    parser.add_argument("--train_queries", help="Path to training query directory")
    parser.add_argument("--results", help="Path to output directory for results")
    parser.add_argument(
        "--model_dir", help="Path to directory for saving/loading models"
    )
    parser.add_argument("--inverted_index", help="Path to inverted index JSON file")
    parser.add_argument("--top_k", type=int, help="Number of top results to return")
    parser.add_argument("--retrain", action="store_true", help="Force retraining")

    args = parser.parse_args()
    Config.update_from_args(args)

    main()
