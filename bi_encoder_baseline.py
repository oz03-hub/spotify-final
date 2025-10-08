#!/usr/bin/env python3
"""
Fast creation of pairwise bi-encoder training data (single-threaded).
"""

import random
from pathlib import Path
import tqdm
import orjson
import util
import os

def create_batch_data():
    with open("dataset/tracks_index.json", "rb") as f:
        all_tracks = orjson.loads(f.read())

    all_ids = set(all_tracks.keys())
    track_name_map = {
        tid: f'{t["track_name"]} {t["artist_name"]} {t["album_name"]}'
        for tid, t in all_tracks.items()
    }

    directory = "dataset/spotify_million_playlist_dataset/data"
    files = list(Path(directory).glob("*.json"))
    for file in tqdm.tqdm(files, desc="Creating paiwise data"):
        with open(file, "rb") as f:
            data = orjson.loads(f.read())

        training_data = {}
        for playlist in data.get("playlists", []):
            pid = playlist["pid"]
            tracks = playlist.get("tracks", [])

            track_ids = {util._extract_id(t["track_uri"]) for t in tracks}
            track_names = [
                f'{t["track_name"]} {t["artist_name"]} {t["album_name"]}' for t in tracks
            ]

            training_data[pid] = {
                "name": playlist.get("name", ""),
                "relevant_track_ids": track_ids,
                "relevant_track_names": track_names,
            }

        results = []
        for pid, pdata in tqdm.tqdm(training_data.items(), desc="Sampling unrelevant tracks"):
            results.append(process_playlist(pid, pdata, all_ids, track_name_map))

        training_data = dict(results)
        out_path = os.path.join("dataset/bi_encoder_batch_data", file.name)
        with open(out_path, "wb") as f:
            f.write(orjson.dumps(training_data, option=orjson.OPT_INDENT_2))
    
    print("\nBatch training data written to dataset/bi_encoder_batch_data/")


def process_playlist(pid, pdata, all_ids, track_name_map):
    """Sample unrelevant tracks and add names."""
    rel = pdata["relevant_track_ids"]
    diff = list(all_ids - rel)
    n = len(rel)
    if len(diff) >= n:
        neg = random.sample(diff, n)
    else:
        neg = diff  # fallback if not enough

    pdata["unrelevant_track_ids"] = neg
    pdata["unrelevant_track_names"] = [track_name_map[i] for i in neg]
    return pid, pdata


if __name__ == "__main__":
    if not os.path.exists("dataset/bi_encoder_batch_data/"):
        create_batch_data()


# """Retrieves songs by encoding both the query and the songs using a bi-encoder model."""

# import json
# import random
# import os
# import tqdm
# import util


# def read_training_tracks(directory: str):
#     files = [os.path.join(directory, f) for f in os.listdir(directory)]

#     trainig_data = {}
#     for file in tqdm.tqdm(files, desc="Reading training data"):
#         with open(file, "r") as f:
#             data = json.load(f)

#         playlists = data["playlists"]
#         for playlist in playlists:
#             playlist_id = playlist["pid"]
#             playlist_name = playlist["name"]
#             track_ids = set(
#                 [util._extract_id(track["track_uri"]) for track in playlist["tracks"]]
#             )

#             track_names = [
#                 " ".join(
#                     [track["track_name"], track["artist_name"], track["album_name"]]
#                 )
#                 for track in playlist["tracks"]
#             ]

#             trainig_data[playlist_id] = {
#                 "name": playlist_name,
#                 "relevant_track_ids": track_ids,
#                 "relevant_track_names": track_names,
#             }

#     return trainig_data


# def sample_unrelevant_tracks(all_track_ids: set, relevant_track_ids: set):
#     unrelevant_track_ids = list(all_track_ids - relevant_track_ids)
#     return random.sample(unrelevant_track_ids, len(relevant_track_ids))


# if __name__ == "__main__":
#     with open("dataset/tracks_index.json") as f:
#         all_tracks = json.load(f)

#     all_track_ids = set(all_tracks.keys())
#     training_data = read_training_tracks(
#         "dataset/spotify_million_playlist_dataset/data"
#     )

#     for playlist_id, playlist_data in tqdm.tqdm(
#         training_data.items(), desc="Sampling unrelevant tracks"
#     ):
#         relevant_track_ids = playlist_data["relevant_track_ids"]
#         unrelevant_track_ids = sample_unrelevant_tracks(
#             all_track_ids, relevant_track_ids
#         )

#         training_data[playlist_id]["unrelevant_track_ids"] = unrelevant_track_ids
#         training_data[playlist_id]["unrelevant_track_names"] = [
#             " ".join(
#                 [
#                     all_tracks[track_id]["track_name"],
#                     all_tracks[track_id]["artist_name"],
#                     all_tracks[track_id]["album_name"],
#                 ]
#             )
#             for track_id in unrelevant_track_ids
#         ]

#     with open("dataset/bi_encoder_training_data.json", "w") as f:
#         json.dump(training_data, f, indent=4)
