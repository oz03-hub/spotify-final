import json
import random
import evaluation
import tqdm

if __name__ == "__main__":
    with open("dataset/spotify_million_playlist_dataset_challenge/challenge_set.json") as f:
        data = json.load(f)

    with open("dataset/tracks_index.json") as f:
        tracks = json.load(f)

    track_ids = list(tracks.keys())
    submission = {}
    for playlist in tqdm.tqdm(data["playlists"]):
        sampled_ids = random.sample(track_ids, 100)
        submission[str(playlist["pid"])] = [tracks[track_id] for track_id in sampled_ids]

    qrels = {}
    for playlist in data["playlists"]:
        qrels[str(playlist["pid"])] = playlist["tracks"]
    
    p_3 = evaluation.precision_at_k(submission, qrels, 3)
    p_10 = evaluation.precision_at_k(submission, qrels, 10)
    p_100 = evaluation.precision_at_k(submission, qrels, 100)
    rr = evaluation.reciprocal_rank(submission, qrels)

    print(f"P@3: {p_3["mean"]:.4f}")
    print(f"P@10: {p_10["mean"]:.4f}")
    print(f"P@100: {p_100["mean"]:.4f}")
    print(f"MRR: {rr["mean"]:.4f}")
