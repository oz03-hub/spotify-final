def _extract_id(uri):
    return uri.split(":")[2]


def _parse_qrel(qrel):
    qrel_tracks = set()
    qrel_artists = set()

    for track in qrel:
        qrel_tracks.add(_extract_id(track["track_uri"]))
        qrel_artists.add(_extract_id(track["artist_uri"]))

    return qrel_tracks, qrel_artists


def precision_at_k(submission: dict[list], qrel: dict[list], k: int) -> dict[float]:
    """Computes precision at K with challenge formula

    Args:
        submission (dict[list]): Submission JSON, pid -> Tracks
        qrel (dict[list]): Qrels JSON, pid -> Tracks
        k (int): K

    Returns:
        dict[float]: pid -> precision
    """

    def _precision_for_pid(sub: list[dict], qrel: list[dict]):
        qrel_tracks, qrel_artists = _parse_qrel(qrel)

        correct = 0
        for track in sub[:k]:
            track_id = _extract_id(track["track_uri"])
            artist_uri = _extract_id(track["artist_uri"])

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

def reciprocal_rank(submission: dict[list], qrel: dict[list]) -> dict[float]:
    """Computes reciprocal rank with challenge formula

    Args:
        submission (dict[list]): Submission JSON, pid -> Tracks
        qrel (dict[list]): Qrels JSON, pid -> Tracks

    Returns:
        dict[float]: pid -> reciprocal rank
    """

    def _reciprocal_rank_for_pid(sub: list[dict], qrel: list[dict]):
        qrel_tracks, _ = _parse_qrel(qrel)

        for i, track in enumerate(sub):
            track_id = _extract_id(track["track_uri"])

            if track_id in qrel_tracks:
                return 1 / (i + 1)
        return 0.0

    rranks = {}
    for pid in submission:
        rranks[pid] = _reciprocal_rank_for_pid(submission[pid], qrel[pid])
    
    rranks["mean"] = sum(rranks.values()) / len(rranks)
    return rranks
