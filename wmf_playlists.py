import os
import random

import numpy as np
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
from pathlib import Path
from tqdm import tqdm
from util import load_corpus, load_queries, get_query_files, save_results
from wmf_baseline import WMFModel
import util
import json

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
    NARROW_DIR = Path("dataset/narrow/train_narrow.json")

    # Model parameters
    N_FACTORS = 200  # Latent factors
    REGULARIZATION = 0.01  # L2 regularization
    ITERATIONS = 50  # Number of ALS iterations
    ALPHA = 30.0  # Confidence weight multiplier

    # Sampling parameters
    K_S = 2 # Number of songs sampled per top K_P playlists
    K_P = 10 # Top K_P most similar playlists retrieved

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

with open("dataset/playlist_path_index.json") as f:
    ppi = json.load(f)

def save_playlists(model: WMFModel, queries, top_k):
    results = {}
    for idx, playlist in enumerate(tqdm(queries, desc="Retrieving")):
        query_text = f"{playlist.get('name', '')} {playlist.get('description', '')}"
        pid = playlist.get("pid")
        # retrieved = [(playlist_id, score), ...]
        retrieved = model.retrieve_playlists(query_text, top_k=top_k)
        results[pid] = retrieved
    return results

def save_songs_and_p_scores(playlist_results: dict):
    results = {}
    with open(Config.NARROW_DIR) as f:
        p_file = json.load(f)

    for pid, similar_tuples in tqdm(playlist_results.items(), desc="Sampling Songs"):
        sampled_songs = []
        for similar_pid, similarity_score in similar_tuples:
            track_uris = p_file[str(similar_pid)]
            # Randomly sample 2 songs (or fewer if playlist has less than 2)
            sample_size = min(Config.K_S, len(track_uris))
            sampled_tracks = random.sample(track_uris, sample_size) if sample_size > 0 else []
            for track in sampled_tracks:
                sampled_songs.append((track, similarity_score))
            
        if (len(sampled_songs)) == 0:
            # for track in sampled_tracks:
            #     print(track)
            # print("SKIP")
            # print("PID", pid)
            continue
        
        scores = [i[1] for i in sampled_songs]
        softmax_scores = softmax(scores)

        for i in range(len(sampled_songs)):
            updated_tuple = (sampled_songs[i][0], softmax_scores[i])
            sampled_songs[i] = updated_tuple

        results[pid] = sampled_songs

    return results

def softmax(x):
    """Compute softmax values for array x."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

def save_songs(playlist_results: dict):
    results = {}
    with open(Config.NARROW_DIR) as f:
        p_file = json.load(f)

    for pid, similar_tuples in tqdm(playlist_results.items(), desc="Sampling Songs"):
        sampled_songs = []
        for similar_pid, similarity_score in similar_tuples:
            track_uris = p_file[str(similar_pid)]
            
            # Randomly sample 2 songs (or fewer if playlist has less than 2)
            sample_size = min(Config.K_S, len(track_uris))
            sampled_tracks = random.sample(track_uris, sample_size) if sample_size > 0 else []

            sampled_songs.extend(sampled_tracks)
        
        results[pid] = sampled_songs

    return results

# def process_queries(model: WMFModel, queries, top_k):
#     """Process all queries and save results."""
#     # Runs and saves the top K_P most similar playlists for each query
#     playlist_results = save_playlists(model, queries, top_k)
#     # Runs and saves the top K_S randomly sampled from the top K_P playlists for each query.
#     song_results = save_songs(playlist_results)

#     final_results = {}
#     idx = 0
#     for pid, top_songs_uris in tqdm(song_results.items(), desc="Processing Song Embeddings"):
#         if not top_songs_uris:
#             continue

#         avg_song_embedding = get_average_track_embedding(model, top_songs_uris)
#         track_recs = recommend_similar_tracks(model, embedding=avg_song_embedding)
#         final_results[pid] = track_recs
        
#         # results is a list of (track_id, score) tuples
#         for pid, songs in song_results.items():
#             # # Printing query text + top 10 retrieved playlist titles
#             if idx % 100 == 0:
#                 pid_file = util.find_playlist_file(pid, ppi)
#                 with open(pid_file) as f:
#                     p_file = json.load(f)
#                 pls = p_file["playlists"]
#                 print()
#                 print("PLAYLIST TITLE", pls[pid % 1000]["name"])
#                 for track_id, track_score in track_recs:
#                     track_data = model.corpus[track_id]
#                     track_name = track_data.get("track_name", "")
#                     print(track_name)
#                 print("NUM TRACKS ", len(track_recs))
#                 print()
#             idx += 1
        
        
    
#     return playlist_results, song_results

def recommend_similar_tracks(self, track_id=None, embedding=None, top_k=100):
    """
    Find similar tracks based on latent embeddings.
    
    Args:
        track_id: Single track ID to find similar tracks for (optional)
        embedding: Pre-computed embedding vector (optional)
        top_k: Number of results to return
    
    Returns:
        List of (track_id, score) tuples
    """
    if self.model is None:
        raise ValueError("Model not trained.")
    
    # Get query embedding from either track_id or provided embedding
    if embedding is not None:
        query_embedding = embedding
        exclude_idx = None
    elif track_id is not None:
        if track_id not in self.track_id_to_idx:
            return []
        exclude_idx = self.track_id_to_idx[track_id]
        query_embedding = self.model.user_factors[exclude_idx]
    else:
        raise ValueError("Must provide either track_id or embedding")
    
    # Compare with all track embeddings
    track_embeddings = self.model.user_factors[:self.num_tracks]
    similarities = track_embeddings.dot(query_embedding)
    
    # Get top-k (excluding the query track if applicable)
    top_k_search = min(top_k + 1 if exclude_idx is not None else top_k, len(similarities))
    top_indices = np.argpartition(-similarities, top_k_search)[:top_k_search]
    top_indices = top_indices[np.argsort(-similarities[top_indices])]
    
    results = []
    for idx in top_indices:
        if exclude_idx is None or idx != exclude_idx:
            results.append((self.track_ids[idx], float(similarities[idx])))
    
    return results[:top_k]

def get_weighted_track_embedding(model, pid, tracks_and_p_scores):
    """
    Get the average embedding for a list of track URIs.
    """
    if model.model is None:
        raise ValueError("Model not trained.")
    
    if not tracks_and_p_scores:
        print("WARNING: No track URIs provided")
        return None
    
    # Build URI lookup if it doesn't exist
    if not hasattr(model, 'track_uri_to_id'):
        print("Building track_uri_to_id lookup...")
        model.track_uri_to_id = {
            track_data['track_uri']: track_id 
            for track_id, track_data in model.corpus.items()
        }
    
    valid_embeddings = []
    scores = []
    
    for tuple in tracks_and_p_scores[pid]:
        track_uri, playlist_score = tuple
        track_id = model.track_uri_to_id.get(track_uri)
        scores.append(playlist_score)
        
        if track_id and track_id in model.track_id_to_idx:
            track_idx = model.track_id_to_idx[track_id]
            embedding = model.model.user_factors[track_idx]
            valid_embeddings.append(embedding)
    
    # print(f"Input URIs: {len(track_uris)}, Valid embeddings: {len(valid_embeddings)}")
    
    if not valid_embeddings:
        print(f"WARNING: No valid embeddings found for URIs: {tracks_and_p_scores[:3]}...")
        return None
    
    # Average the embeddings
    avg_embedding = np.average(valid_embeddings, weights=scores, axis=0)
    return avg_embedding

def get_average_track_embedding(model, track_uris):
    """
    Get the average embedding for a list of track URIs.
    """
    if model.model is None:
        raise ValueError("Model not trained.")
    
    if not track_uris:
        print("WARNING: No track URIs provided")
        return None
    
    # Build URI lookup if it doesn't exist
    if not hasattr(model, 'track_uri_to_id'):
        print("Building track_uri_to_id lookup...")
        model.track_uri_to_id = {
            track_data['track_uri']: track_id 
            for track_id, track_data in model.corpus.items()
        }
    
    valid_embeddings = []
    
    for track_uri in track_uris:
        track_id = model.track_uri_to_id.get(track_uri)
        
        if track_id and track_id in model.track_id_to_idx:
            track_idx = model.track_id_to_idx[track_id]
            embedding = model.model.user_factors[track_idx]
            valid_embeddings.append(embedding)
    
    # print(f"Input URIs: {len(track_uris)}, Valid embeddings: {len(valid_embeddings)}")
    
    if not valid_embeddings:
        print(f"WARNING: No valid embeddings found for URIs: {track_uris[:3]}...")
        return None
    
    # Average the embeddings
    avg_embedding = np.mean(valid_embeddings, axis=0)
    return avg_embedding


def process_queries(model: WMFModel, queries, top_kp):
    """Process all queries and save results."""
    playlist_results = save_playlists(model, queries, top_kp)
    song_results = save_songs(playlist_results)
    songs_and_p_results = save_songs_and_p_scores(playlist_results)

    final_results = {}
    final_weighted_results = {}
    idx = 0
    
    for pid, top_songs_uris in tqdm(song_results.items(), desc="Processing Song Embeddings"):
        if not top_songs_uris:
            # print("PRIOR PID", pid)
            continue

        # DEBUG: Print the input songs for this playlist
        if idx % 100 == 0:
            pid_file = util.find_playlist_file(pid, ppi)
            with open(pid_file) as f:
                p_file = json.load(f)
            pls = p_file["playlists"]
            print()
            print("PLAYLIST TITLE", pls[pid % 1000]["name"])
            # print(f"Input songs (first 5): {top_songs_uris[:5]}")
        
        avg_song_embedding = get_average_track_embedding(model, top_songs_uris)
        weighted_avg_song_embedding = get_weighted_track_embedding(model, pid, songs_and_p_results)
        
        if avg_song_embedding is None:
            print(f"Skipping playlist {pid} - no valid embeddings")
            continue
        
        # DEBUG: Check if embedding is different
        if idx % 100 == 0:
            print(f"Embedding hash: {hash(avg_song_embedding.tobytes())}")
        
        track_recommendations = recommend_similar_tracks(
            model, 
            embedding=avg_song_embedding, 
            top_k=100
        )
        weighted_track_recommendations = recommend_similar_tracks(
            model, 
            embedding=weighted_avg_song_embedding, 
            top_k=100
        )
        
        final_results[pid] = track_recommendations
        final_weighted_results[pid] = weighted_track_recommendations

        # Debug printing
        if idx % 100 == 0:
            print(f"Top 20 recommendations:")
            for track_id, track_score in track_recommendations[:20]:
                track_data = model.corpus[track_id]
                track_name = track_data.get("track_name", "")
                print(f"  {track_name} (score: {track_score:.4f})")
            print()
        
        idx += 1
    
    return playlist_results, song_results, final_results, final_weighted_results

# def debug_process_queries(model: WMFModel, queries, top_k):
#     """Process all queries and print results (K_S songs randomly sampled for each top K_P playlists)."""
#     results = {}
#     for idx, playlist in enumerate(tqdm(queries, desc="Retrieving")):
#         query_text = f"{playlist.get('name', '')} {playlist.get('description', '')}"
#         pid = playlist.get("pid")
#         # retrieved = [(playlist_id, score), ...]
#         retrieved = model.retrieve_playlists(query_text, top_k=top_k)
#         # results[pid] = retrieved

#         # Randomly sample 2 songs from each retrieved playlist
#         sampled_results = []
#         sampled_songs = []
#         for ret_pid, ret_score in retrieved:
#             pid_file = util.find_playlist_file(ret_pid, ppi)
#             with open(pid_file) as f:
#                 p_file = json.load(f)
#             pls = p_file["playlists"]
            
#             # Get the playlist tracks
#             playlist_data = pls[ret_pid % 1000]
#             track_names = []
#             track_data = playlist_data.get("tracks", [])
#             for track in track_data:
#                 track_names.append(track["track_name"])
            
#             # Randomly sample 2 songs (or fewer if playlist has less than 2)
#             sample_size = min(Config.K_S, len(track_names))
#             sampled_tracks = random.sample(track_names, sample_size) if sample_size > 0 else []
            
#             sampled_songs.extend(sampled_tracks)
#         results[pid] = sampled_results

#         # # Printing query text + top 10 retrieved playlist titles
#         # if idx % 100 == 0:
#         #     print()
#         #     print(f"Query TEXT: {query_text}")
#         #     titles = []
#         #     for ret_pid, ret_score in retrieved[:10]:
#         #         pid_file = util.find_playlist_file(ret_pid, ppi)
#         #         with open(pid_file) as f:
#         #             p_file = json.load(f)
#         #         pls = p_file["playlists"]

#         #         print(pls[ret_pid % 1000]["name"])
#         #     print()
#     # results = "query playlist": [(playlist_id, score), ...] sorted in descending order
#     return results

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

        split_result_dir_playlist = Config.RESULTS_DIR / "playlist" / split
        split_result_dir_playlist.mkdir(parents=True, exist_ok=True)

        split_result_dir_unweighted = Config.RESULTS_DIR / "unweighted" / split
        split_result_dir_unweighted.mkdir(parents=True, exist_ok=True)

        split_result_dir_weighted = Config.RESULTS_DIR / "wmf_weighted" / split
        split_result_dir_weighted.mkdir(parents=True, exist_ok=True)

        query_files = get_query_files(split_dir)
        print(f"\n{'=' * 70}")
        print(f"Processing {split} split ({len(query_files)} files)")
        print("=" * 70)

        for query_file in query_files:
            print(f"\nProcessing: {query_file.name}")
            queries = load_queries(query_file)
            playlist_results, song_results, final_results, final_weighted_results = process_queries(model, queries, Config.K_P)
            stripped_results = {}
            stripped_weighted_results = {}

            for pid, ranked_songs_list in final_results.items():
                stripped_results[pid] = []
                track_ids = [t[0] for t in ranked_songs_list]
                for track_id in track_ids:
                    formatted_track = {}
                    formatted_track["track_uri"] = corpus[track_id]["track_uri"]
                    formatted_track["artist_uri"] = corpus[track_id]["artist_uri"]
                    stripped_results[pid].append(formatted_track)
            
            for pid, ranked_songs_list in final_weighted_results.items():
                stripped_weighted_results[pid] = []
                track_ids = [t[0] for t in ranked_songs_list]
                for track_id in track_ids:
                    formatted_track = {}
                    formatted_track["track_uri"] = corpus[track_id]["track_uri"]
                    formatted_track["artist_uri"] = corpus[track_id]["artist_uri"]
                    stripped_weighted_results[pid].append(formatted_track)

            #Saves query PID + Similar Playlists and Similarity Score
            save_results(playlist_results, query_file.name, split_result_dir_playlist)
            save_results(stripped_results, query_file.name, split_result_dir_unweighted)
            save_results(stripped_weighted_results, query_file.name, split_result_dir_weighted)

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