import math
from util import extract_id

def _parse_qrel(qrel):
    qrel_tracks = set()
    qrel_artists = set()

    for track in qrel:
        qrel_tracks.add(extract_id(track["track_uri"]))
        qrel_artists.add(extract_id(track["artist_uri"]))

    return qrel_tracks, qrel_artists


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

def evaluation_report(submission: dict[list], qrel_obj) -> dict[float]:
    """Generate P@3, P@5, P@10, P@R, RR report

    Args:
        submission (dict[list]): submission JSON, pid -> Tracks
        qrel (dict[list]): qrels JSON

    Returns:
        dict[float]: evaluation report
    """

    qrel = {}
    for playlist in qrel_obj["playlists"]:
        qrel[str(playlist["pid"])] = playlist["tracks"]


    p5 = precision_at_k(submission, qrel, 5)
    p10 = precision_at_k(submission, qrel, 10)
    p100 = precision_at_k(submission, qrel, 100)
    pr = r_precision(submission, qrel)

    rr = reciprocal_rank(submission, qrel)

    ndcg_at_5 = ndcg_at_k(submission, qrel, 5)
    ndcg_at_10 = ndcg_at_k(submission, qrel, 10)
    ndcg_at_100 = ndcg_at_k(submission, qrel, 100)

    report = {"P@5": p5, "P@10": p10, "P@100": p100, "P@R": pr, "RR": rr, "NDCG@5": ndcg_at_5, "NDCG@10": ndcg_at_10, "NDCG@100": ndcg_at_100}

    return report
