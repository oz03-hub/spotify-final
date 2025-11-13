import os
import json
import tqdm
import orjson
from pathlib import Path
from util import extract_id, tokenize
from collections import Counter

if __name__ == "__main__":
    dataset_path = Path("dataset")
    splits = ["train", "val", "test"]

    playlist_path_index = {}
    tracks_index = {}
    playlist_inverted_index = {}
    playlist_inverted_index["DOCID_MAP"] = []
    playlist_inverted_index["TOTAL_DOCS"] = 0
    playlist_inverted_index["DOC_LENGTHS"] = []

    for split in splits:
        split_path = dataset_path / split
        files = os.listdir(split_path)
        split_avg_len = 0
        for file in tqdm.tqdm(files, desc=f"Processing {split} split"):
            file_path = split_path / file
            min_pid = file.split("-")[0]
            max_pid = file.split("-")[1].split(".json")[0]
            bucket_key = f"{min_pid}_{max_pid}"
            playlist_path_index[bucket_key] = str(file_path)
            with open(file_path, "r") as f:
                data = orjson.loads(f.read())

                for playlist in data["playlists"]:
                    p_tokens = Counter(tokenize(f"{playlist['name']} {playlist.get('description', '')}"))
                    playlist_inverted_index["DOCID_MAP"].append(playlist["pid"])
                    playlist_inverted_index["DOC_LENGTHS"].append(sum(p_tokens.values()))
                    playlist_inverted_index["TOTAL_DOCS"] += 1

                    for token, tf in p_tokens.items():
                        if token not in playlist_inverted_index:
                            playlist_inverted_index[token] = list()
                        playlist_inverted_index[token].append((playlist["pid"], tf))

                    split_avg_len += len(playlist["tracks"])
                    for track in playlist["tracks"]:
                        track_id = extract_id(track["track_uri"])
                        if track_id not in tracks_index:
                            tracks_index[track_id] = track
                            tracks_index[track_id]["extended_text"] = " ".join(
                                tokenize(
                                    track["track_name"]
                                    + " "
                                    + track["artist_name"]
                                    + " "
                                    + track["album_name"]
                                )
                            )
                            tracks_index[track_id]["pf"] = 1
                            del tracks_index[track_id]["pos"]

                        if split == "train":
                            extended_text = " ".join(
                                tokenize(
                                    playlist["name"]
                                    + " "
                                    + playlist.get("description", "")
                                )
                            )
                            tracks_index[track_id]["extended_text"] += (
                                " " + extended_text
                            )
                            tracks_index[track_id]["pf"] += 1

        print(
            f"Average number of playlists per file in {split} split: {split_avg_len / len(files) * 1000}"
        )

    with open(dataset_path / "track_corpus.json", "w") as f:
        json.dump(tracks_index, f, indent=2)

    with open(dataset_path / "playlist_path_index.json", "w") as f:
        json.dump(playlist_path_index, f, indent=2)

    with open(dataset_path / "playlist_inverted_index.json", "w") as f:
        json.dump(playlist_inverted_index, f, indent=2)

    inverted_index = {}
    inverted_index["DOCID_MAP"] = []
    inverted_index["TOTAL_DOCS"] = len(tracks_index)
    inverted_index["DOC_LENGTHS"] = []

    for docid, (track_id, track) in tqdm.tqdm(enumerate(tracks_index.items())):
        tokens = Counter(tokenize(track["extended_text"]))

        inverted_index["DOCID_MAP"].append(track_id)
        inverted_index["DOC_LENGTHS"].append(sum(tokens.values()))

        for token, tf in tokens.items():
            if token not in inverted_index:
                inverted_index[token] = list()
            inverted_index[token].append((docid, tf))

    with open(dataset_path / "inverted_index.json", "w") as f:
        json.dump(inverted_index, f, indent=2)
    print(
        f"Inverted index built and saved. {len(inverted_index)-2} unique tokens, {inverted_index['TOTAL_DOCS']} Documents indexed."
    )
