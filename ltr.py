import json
from pathlib import Path
from tqdm import tqdm
import os
import orjson
import numpy as np
import xgboost as xgb
import argparse
from util import load_corpus, get_query_files, save_results


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Centralized configuration for paths and parameters."""

    # Default paths
    WORKSPACE_DIR = Path("dataset")
    INVERTED_INDEX = WORKSPACE_DIR / "inverted_index.json"
    CORPUS_FILE = WORKSPACE_DIR / "track_corpus.json"
    RERANKER_TRAIN_DIR = WORKSPACE_DIR / "reranker_dataset" / "train"
    RERANKER_VAL_DIR = WORKSPACE_DIR / "reranker_dataset" / "val"
    RERANKER_TEST_DIR = WORKSPACE_DIR / "reranker_dataset" / "test"

    RESULTS_DIR = WORKSPACE_DIR / "results" / "ltr_baseline"
    TEST_RESULTS_DIR = RESULTS_DIR / "test"
    VAL_RESULTS_DIR = RESULTS_DIR / "val"

    # Model parameters
    TOP_K = 100
    MODEL_PATH = Path("model") / "ltr_ranker.xgb"

    # Training parameters
    ETA = 0.01
    MAX_DEPTH = 6
    NUM_BOOST_ROUND = 400
    DEVICE = "cuda"

    EVAL_ONLY = False

    @classmethod
    def update_from_args(cls, args):
        """Update configuration from command line arguments."""
        if args.corpus:
            cls.CORPUS_FILE = Path(args.corpus)
        if args.train_dir:
            cls.RERANKER_TRAIN_DIR = Path(args.train_dir)
        if args.val_dir:
            cls.RERANKER_VAL_DIR = Path(args.val_dir)
        if args.test_dir:
            cls.RERANKER_TEST_DIR = Path(args.test_dir)
        if args.results:
            cls.RESULTS_DIR = Path(args.results)
            cls.TEST_RESULTS_DIR = cls.RESULTS_DIR / "test"
            cls.VAL_RESULTS_DIR = cls.RESULTS_DIR / "val"
        if args.inverted_index:
            cls.INVERTED_INDEX = Path(args.inverted_index)
        if args.top_k:
            cls.TOP_K = args.top_k
        if args.model_path:
            cls.MODEL_PATH = args.model_path

        # Training parameters
        if args.eta:
            cls.ETA = args.eta
        if args.max_depth:
            cls.MAX_DEPTH = args.max_depth
        if args.num_boost_round:
            cls.NUM_BOOST_ROUND = args.num_boost_round
        if args.device:
            cls.DEVICE = args.device

        if args.eval_only:
            cls.EVAL_ONLY = True

        # Ensure results directory exists
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.VAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_PATH.parent.mkdir(exist_ok=True)


class PlaylistLTRDataset:
    """
    Wrapper to iterate through multiple JSON files, each with multiple query groups.
    Produces an XGBoost DMatrix ready for rank:pairwise training.
    """

    def __init__(self, data_dir, feature_keys=None, exclude_features=None):
        self.data_dir = data_dir
        self.files = sorted(
            [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".json")
            ]
        )

        # Default: use all numeric fields except excluded ones
        self.exclude_features = set(exclude_features or ["track_id", "label"])
        self.feature_keys = feature_keys

    def _load_file(self, file_path):
        """Load and flatten all queries in a single file."""
        with open(file_path, "r") as f:
            data = json.load(f)

        X_list, y_list, group_list = [], [], []

        for qid, docs in data.items():
            if not docs:
                continue

            # auto infer feature keys if not provided
            if self.feature_keys is None:
                self.feature_keys = [
                    k for k in docs[0].keys() if k not in self.exclude_features
                ]

            X = np.array([[d[k] for k in self.feature_keys] for d in docs], dtype=float)
            y = np.array([d["label"] for d in docs], dtype=float)

            X_list.append(X)
            y_list.append(y)
            group_list.append(len(docs))

        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)
        return X_all, y_all, group_list

    def to_dmatrix(self):
        """Combine all JSON files into a single DMatrix."""
        X_all, y_all, group_all = [], [], []

        for fpath in self.files:
            try:
                X, y, group = self._load_file(fpath)
            except Exception as e:
                print(e)
                print(fpath)
                continue

            X_all.append(X)
            y_all.append(y)
            group_all.extend(group)

        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all)

        dtrain = xgb.DMatrix(X_all, label=y_all)
        dtrain.set_group(group_all)
        return dtrain


def train_model():
    print("Training")
    dataset = PlaylistLTRDataset(
        data_dir=Config.RERANKER_TRAIN_DIR,
        exclude_features=["track_id", "label"],
    )

    dtrain = dataset.to_dmatrix()

    params = {
        "objective": "rank:map",
        "eval_metric": "ndcg",
        "eta": Config.ETA,
        "max_depth": Config.MAX_DEPTH,
        "tree_method": "hist",
        "device": "cuda",
    }

    model = xgb.train(params, dtrain, num_boost_round=Config.NUM_BOOST_ROUND)
    model.save_model(Config.MODEL_PATH)
    return model


# ============================================================================
# RESULTS PROCESSING
# ============================================================================
def process_queries(model, queries, top_k):
    EXCLUDE = {"track_id", "label"}

    results = {}

    for pid, docs in tqdm(queries.items(), desc="Reranking"):
        if not docs:
            results[pid] = []
            continue

        feature_keys = [k for k in docs[0].keys() if k not in EXCLUDE]

        X = np.array([[doc[k] for k in feature_keys] for doc in docs], dtype=float)
        dmatrix = xgb.DMatrix(X)

        scores = model.predict(dmatrix)

        track_scores = [
            (doc["track_id"], float(score)) for doc, score in zip(docs, scores)
        ]
        track_scores.sort(key=lambda x: x[1], reverse=True)
        results[pid] = track_scores[:top_k]
    return results


def main():
    if not Config.EVAL_ONLY:
        model = train_model()
    else:
        model = xgb.Booster()
        model.load_model(Config.MODEL_PATH)

    corpus = load_corpus(Config.CORPUS_FILE)

    for query_dir, results_dir in [
        (Config.RERANKER_TEST_DIR, Config.TEST_RESULTS_DIR),
        (Config.RERANKER_VAL_DIR, Config.VAL_RESULTS_DIR),
    ]:
        query_files = get_query_files(query_dir)
        for query_file in query_files:
            with open(query_file) as f:
                queries = orjson.loads(f.read())

            results = process_queries(model, queries, Config.TOP_K)
            for pid in results:
                replacement_tracks = []
                for track_id, _ in results[pid]:
                    track = corpus[track_id]
                    eval_track = {
                        "track_uri": track["track_uri"],
                        "artist_uri": track["artist_uri"],
                    }
                    replacement_tracks.append(eval_track)
                results[pid] = replacement_tracks

            save_results(results, query_file.name, results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate Learning-to-Rank model for playlist continuation"
    )

    # Data paths
    parser.add_argument("--corpus", type=str, help="Path to corpus file")
    parser.add_argument("--train-dir", type=str, help="Path to training data directory")
    parser.add_argument("--val-dir", type=str, help="Path to validation data directory")
    parser.add_argument("--test-dir", type=str, help="Path to test data directory")
    parser.add_argument("--results", type=str, help="Path to results directory")
    parser.add_argument("--inverted-index", type=str, help="Path to inverted index")

    # Model parameters
    parser.add_argument("--model-path", type=str, help="Path to save/load model")
    parser.add_argument("--top-k", type=int, help="Number of top results to return")

    # Training parameters
    parser.add_argument("--eta", type=float, help="Learning rate")
    parser.add_argument("--max-depth", type=int, help="Maximum tree depth")
    parser.add_argument("--num-boost-round", type=int, help="Number of boosting rounds")
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda"], help="Device to use for training"
    )

    parser.add_argument(
        "--eval-only", action="store_true", help="Only run evaluation on val/test sets"
    )

    args = parser.parse_args()

    # Update configuration from arguments
    Config.update_from_args(args)

    main()
