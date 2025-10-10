import json
import random
import tqdm
import os
from pathlib import Path

if __name__ == "__main__":
    workspace_dir = Path("dataset")
    test_dir = workspace_dir / "test"
    results_dir = workspace_dir / "results" / "random_baseline"
    results_dir.mkdir(parents=True, exist_ok=True)

    test_files = os.listdir(test_dir)
    test_file = [test_dir / f for f in test_files]

    with open(workspace_dir / "tracks_index.json") as f:
        tracks = json.load(f)
    track_ids = list(tracks.keys())

    for f in test_file:
        with open(f) as infile:
            data = json.load(infile)

        submission = {}
        for playlist in tqdm.tqdm(data["playlists"]):
            sampled_ids = random.sample(track_ids, len(playlist["tracks"]))
            submission[str(playlist["pid"])] = [tracks[track_id] for track_id in sampled_ids]

        output_path = results_dir / f.name
        with open(output_path, "w") as outfile:
            json.dump(submission, outfile)
        print(f"Results written to: {output_path}")
