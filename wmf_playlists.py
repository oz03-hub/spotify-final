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
    K_FINAL = 100 # Number of songs returned in the final playlists

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

def get_top_kp_playlists(model: WMFModel, queries):
    '''Given seed query playlist title names, retrieve the TOP_KP similar playlists'''
    results = {}
    for idx, playlist in enumerate(tqdm(queries, desc="Retrieving")):
        query_text = f"{playlist.get('name', '')} {playlist.get('description', '')}"
        pid = playlist.get("pid")
        # retrieved = [(playlist_id, score), ...]
        retrieved = model.retrieve_playlists(query_text, top_k=Config.K_P)
        results[pid] = retrieved
    return results

def get_songs_and_weights(playlist_results: dict):
    '''Given each seed query playlist title and its retrieved similar playlists and respective similarity scores, 
        randomly sample K_S # of songs per playlist and assign a weight based on the similarity score of the playlist it came from.
        Returns a dict, where the key is the seed playlist pid and the value is a list of tuples (track_uri, weight)'''
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

def get_songs(playlist_results: dict):
    '''Given the seed query playlist title and its retrieved similar playlists and respective similarity scores.
    Returns a dict, where the key is the seed playlist pid and the value is a list of track_uri's'''
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

def recommend_similar_tracks(self, embedding=None):
    """
    Find similar tracks based on a latent track embedding.
    
    Args:
        embedding: Pre-computed embedding vector
    
    Returns:
        List of K_FINAL # of (track_id, score) tuples
    """
    if self.model is None:
        raise ValueError("Model not trained.")
    
    # Ensures embedding is passed in, else raise a ValueError
    if embedding is not None:
        query_embedding = embedding
        exclude_idx = None
    else:
        raise ValueError("Must provide an embedding")
    
    # Compare with all track embeddings
    track_embeddings = self.model.user_factors[:self.num_tracks]
    similarities = track_embeddings.dot(query_embedding)
    
    # Get top-k
    top_k_search = min(Config.K_FINAL, len(similarities))
    top_indices = np.argpartition(-similarities, top_k_search)[:top_k_search]
    top_indices = top_indices[np.argsort(-similarities[top_indices])]
    
    results = []
    for idx in top_indices:
        results.append((self.track_ids[idx], float(similarities[idx])))
    
    return results[:Config.K_FINAL]

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
    
    if not valid_embeddings:
        print(f"WARNING: No valid embeddings found for URIs: {track_uris[:3]}...")
        return None
    
    # Average the embeddings
    avg_embedding = np.mean(valid_embeddings, axis=0)
    return avg_embedding


def process_queries(model: WMFModel, queries):
    """Process all queries and save results. Saves results in three different subdirectories. 
        playlist saves the top kp similar playlists to the input playlist name
        unweighted saves the top k_final songs given the input playlist name when song embeddings are averaged
        wmf_weighted saves the top k_final songs given the input playlist name when song embeddings are averaged with weights"""

    playlist_results = get_top_kp_playlists(model, queries)
    song_results = get_songs(playlist_results)
    songs_and_weights_results = get_songs_and_weights(playlist_results)

    final_unweighted_results = {}
    final_weighted_results = {}
    
    for pid, top_songs_uris in tqdm(song_results.items(), desc="Processing Song Embeddings"):
        if not top_songs_uris:
            continue
        
        avg_song_embedding = get_average_track_embedding(model, top_songs_uris)
        weighted_avg_song_embedding = get_weighted_track_embedding(model, pid, songs_and_weights_results)
        
        if avg_song_embedding is None:
            print(f"Skipping playlist {pid} - no valid embeddings")
            continue
        
        track_recommendations = recommend_similar_tracks(
            model, 
            embedding=avg_song_embedding
        )
        weighted_track_recommendations = recommend_similar_tracks(
            model, 
            embedding=weighted_avg_song_embedding
        )
        
        final_unweighted_results[pid] = track_recommendations
        final_weighted_results[pid] = weighted_track_recommendations
    
    return playlist_results, song_results, final_unweighted_results, final_weighted_results

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

        split_result_dir_unweighted = Config.RESULTS_DIR / "wmf_unweighted" / split
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
            playlist_results, song_results, final_results, final_weighted_results = process_queries(model, queries)
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