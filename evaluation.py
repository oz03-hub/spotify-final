import math
from util import extract_id, tokenize

def _parse_qrel(qrel):
    qrel_tracks = set()
    qrel_artists = set()

    for track in qrel:
        qrel_tracks.add(extract_id(track["track_uri"]))
        qrel_artists.add(extract_id(track["artist_uri"]))

    return qrel_tracks, qrel_artists

def recall_at_k(submission: dict[list], qrel: dict[list], k: int) -> dict[float]:
    def _recall_for_pid(sub: list[dict], qrel: list[dict]):
        qrel_tracks, qrel_artists = _parse_qrel(qrel)

        correct = 0
        for track in sub[:k]:
            track_id = extract_id(track["track_uri"])

            if track_id in qrel_tracks:
                correct += 1
        return correct / len(qrel_tracks)

    precisions = {}
    for pid in submission:
        precisions[pid] = _recall_for_pid(submission[pid], qrel[pid])

    precisions["mean"] = sum(precisions.values()) / len(precisions)
    return precisions

def precision_at_k(submission: dict[list], qrel: dict[list], k: int) -> dict[float]:
    """Computes precision at K with challenge formula

    Args:
        submission (dict[list]): Submission JSON, pid -> ids
        qrel (dict[list]): Qrels JSON, pid -> tracks
        k (int): K

    Returns:
        dict[float]: pid -> precision
    """

    def _precision_for_pid(sub: list[dict], qrel: list[dict]):
        qrel_tracks, qrel_artists = _parse_qrel(qrel)

        correct = 0
        for track in sub[:k]:
            track_id = extract_id(track["track_uri"])
            artist_uri = extract_id(track["artist_uri"])

            if track_id in qrel_tracks:
                correct += 1
            elif artist_uri in qrel_artists:
                correct += 0.25
        return correct / k

    precisions = {}
    for pid in submission:
        precisions[pid] = _precision_for_pid(submission[pid], qrel[pid])

    precisions["mean"] = sum(precisions.values()) / len(precisions)
    return precisions


def r_precision(submission: dict[list], qrel: dict[list]) -> dict[float]:
    """Computes R-Precision with challenge formula

    Args:
        submission (dict[list]): Submission JSON, pid -> Tracks
        qrel (dict[list]): Qrels JSON, pid -> Tracks

    Returns:
        dict[float]: pid -> R-Precision
    """

    def _r_precision_for_pid(sub: list[dict], qrel: list[dict]):
        qrel_tracks, qrel_artists = _parse_qrel(qrel)
        R = len(qrel_tracks)

        correct = 0
        for track in sub[:R]:
            track_id = extract_id(track["track_uri"])
            artist_uri = extract_id(track["artist_uri"])

            if track_id in qrel_tracks:
                correct += 1
            elif artist_uri in qrel_artists:
                correct += 0.25
        return correct / R if R > 0 else 0.0

    r_precisions = {}
    for pid in submission:
        r_precisions[pid] = _r_precision_for_pid(submission[pid], qrel[pid])

    r_precisions["mean"] = sum(r_precisions.values()) / len(r_precisions)
    return r_precisions


def reciprocal_rank(submission: dict[list], qrel: dict[list]) -> dict[float]:
    """Computes reciprocal rank with challenge formula

    Args:
        submission (dict[list]): Submission JSON, pid -> ids
        qrel (dict[list]): Qrels JSON, pid -> tracks

    Returns:
        dict[float]: pid -> reciprocal rank
    """

    def _reciprocal_rank_for_pid(sub: list[dict], qrel: list[dict]):
        qrel_tracks, _ = _parse_qrel(qrel)

        for i, track in enumerate(sub):
            track_id = extract_id(track["track_uri"])

            if track_id in qrel_tracks:
                return 1 / (i + 1)
        return 0.0

    rranks = {}
    for pid in submission:
        rranks[pid] = _reciprocal_rank_for_pid(submission[pid], qrel[pid])

    rranks["mean"] = sum(rranks.values()) / len(rranks)
    return rranks

def ndcg_at_k(submission: dict[list], qrel: dict[list], k: int) -> dict[float]:
    """Computes NDCG at K

    Args:
        submission (dict[list]): Submission JSON, pid -> ids
        qrel (dict[list]): Qrels JSON, pid -> tracks
        k (int): K

    Returns:
        dict[float]: pid -> NDCG
    """

    def _dcg(relevances: list[float]) -> float:
        dcg = 0.0
        for i, rel in enumerate(relevances):
            dcg += (2**rel - 1) / math.log2(i + 2)  # i + 2 because i starts from 0
        return dcg

    def _ndcg_for_pid(sub: list[dict], qrel: list[dict]):
        qrel_tracks, qrel_artists = _parse_qrel(qrel)
        relevances = []

        for track in sub[:k]:
            track_id = extract_id(track["track_uri"])
            artist_id = extract_id(track["artist_uri"])

            if track_id in qrel_tracks:
                relevances.append(1.0)
            elif artist_id in qrel_artists:
                relevances.append(0.25)
            else:
                relevances.append(0.0)

        dcg = _dcg(relevances)
        idcg = _dcg([1.0 for _ in range(min(len(qrel_tracks), k))]) # ideal rank is 1.0 for all ranks

        return dcg / idcg if idcg > 0 else 0.0

    ndcgs = {}
    for pid in submission:
        ndcgs[pid] = _ndcg_for_pid(submission[pid], qrel[pid])

    ndcgs["mean"] = sum(ndcgs.values()) / len(ndcgs)
    return ndcgs

def feature_overlap_at_k(
    submission: dict[list],
    qrel: dict[list],
    playlist_metadata: dict,
    track_corpus: dict,
    k: int,
) -> dict[float]:
    """Computes average feature overlap between playlist and retrieved tracks at K
    
    This measures how well the retrieved tracks match the playlist's genre/feature context.
    Higher scores indicate better contextual matching.

    Args:
        submission (dict[list]): Submission JSON, pid -> tracks
        qrel (dict[list]): Qrels JSON (used only for getting PIDs)
        playlist_metadata (dict): Playlist metadata with features/genres
        track_corpus (dict): Track corpus with accumulated features
        k (int): Number of top tracks to evaluate

    Returns:
        dict[float]: pid -> average feature overlap score
    """

    def _feature_overlap_for_pid(sub: list[dict], pid: str):
        # Get playlist features
        pid_str = str(pid)
        labels = playlist_metadata.get(pid_str, {}).get("label", {})
        genres = labels.get("genres", [])
        playlist_features = labels.get("features", [])
        
        # Normalize to lowercase for comparison
        all_playlist_features = set(tokenize(" ".join(genres + playlist_features)))

        # If playlist has no features, return 0
        if not all_playlist_features:
            return 0.0
        
        # Compute overlap for each retrieved track
        overlaps = []
        for track in sub[:k]:
            track_id = extract_id(track["track_uri"])
            
            # Get track features from corpus
            if track_id in track_corpus:
                track_features_set = set(track_corpus[track_id].get("features", []))
                
                # Compute Jaccard similarity (intersection / union)
                if track_features_set:
                    intersection = all_playlist_features.intersection(track_features_set)
                    # Use intersection / playlist_features to measure how well track matches playlist
                    overlap = len(intersection) / len(all_playlist_features)
                else:
                    overlap = 0.0
            else:
                overlap = 0.0
            
            overlaps.append(overlap)
        
        # Return average overlap across top-k tracks
        return sum(overlaps) / len(overlaps) if overlaps else 0.0

    feature_overlaps = {}
    for pid in submission:
        feature_overlaps[pid] = _feature_overlap_for_pid(submission[pid], pid)

    feature_overlaps["mean"] = sum(feature_overlaps.values()) / len(feature_overlaps) if feature_overlaps else 0.0
    return feature_overlaps


def evaluation_report(
    submission: dict[list],
    qrel_obj,
    playlist_metadata,
    track_corpus,
) -> dict[float]:
    """Generate comprehensive evaluation report including feature overlap metrics

    Args:
        submission (dict[list]): submission JSON, pid -> Tracks
        qrel_obj: qrels JSON object with playlists
        playlist_metadata_path (str): Path to playlist metadata JSON file
        track_corpus_path (str): Path to track corpus JSON file

    Returns:
        dict[float]: evaluation report with all metrics
    """

    qrel = {}
    for playlist in qrel_obj["playlists"]:
        qrel[str(playlist["pid"])] = playlist["tracks"]

    # Load metadata and corpus for feature overlap computation
    try:        
        # Compute feature overlap metrics
        feature_overlap_10 = feature_overlap_at_k(submission, qrel, playlist_metadata, track_corpus, 10)
        # feature_overlap_100 = feature_overlap_at_k(submission, qrel, playlist_metadata, track_corpus, 100)
    except FileNotFoundError as e:
        print(f"Warning: Could not load metadata/corpus files for feature overlap: {e}")
        feature_overlap_10 = {"mean": 0.0}

    # Standard metrics
    p5 = precision_at_k(submission, qrel, 5)
    p10 = precision_at_k(submission, qrel, 10)
    p100 = precision_at_k(submission, qrel, 100)
    pr = r_precision(submission, qrel)

    rr = reciprocal_rank(submission, qrel)

    ndcg_at_5 = ndcg_at_k(submission, qrel, 5)
    ndcg_at_10 = ndcg_at_k(submission, qrel, 10)
    ndcg_at_100 = ndcg_at_k(submission, qrel, 100)

    report = {
        "P@5": p5,
        "P@10": p10,
        "P@100": p100,
        "P@R": pr,
        "RR": rr,
        "NDCG@5": ndcg_at_5,
        "NDCG@10": ndcg_at_10,
        "NDCG@100": ndcg_at_100,
        "FeatureOverlap@10": feature_overlap_10,
        # "FeatureOverlap@100": feature_overlap_100
    }

    return report
