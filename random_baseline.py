import json
import random
import evaluation
import tqdm

if __name__ == "__main__":
    with open("dataset/spotify_million_playlist_dataset_challenge/challenge_set_filtered.json") as f:
        data = json.load(f)

    with open("dataset/tracks_index.json") as f:
        tracks = json.load(f)

    track_ids = list(tracks.keys())
    submission = {}
    for playlist in tqdm.tqdm(data["playlists"]):
        sampled_ids = random.sample(track_ids, len(playlist["tracks"]))
        submission[str(playlist["pid"])] = [tracks[track_id] for track_id in sampled_ids]

    qrels = {}
    for playlist in data["playlists"]:
        qrels[str(playlist["pid"])] = playlist["tracks"]
    
    report = evaluation.evaluation_report(submission, qrels)
    print(report["P@3"]["mean"], report["P@5"]["mean"], report["P@10"]["mean"], report["P@100"]["mean"], report["P@R"]["mean"], report["RR"]["mean"])
    