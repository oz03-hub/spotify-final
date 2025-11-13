import orjson
import json
from pathlib import Path
import os
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def extract_id(uri):
    return uri.split(":")[2]


# ============================================================================
# DATA LOADING
# ============================================================================
def load_corpus(corpus_file):
    """Load corpus from JSON file."""
    print(f"Loading corpus from: {corpus_file}")
    with open(corpus_file, "r", encoding="utf-8") as f:
        return orjson.loads(f.read())


def load_queries(query_file):
    """Load queries from JSON file."""
    with open(query_file, "r", encoding="utf-8") as f:
        return orjson.loads(f.read())["playlists"]


def get_query_files(queries_dir):
    """Get all query files from directory."""
    queries_path = Path(queries_dir)
    return [queries_path / f for f in os.listdir(queries_path) if f.endswith(".json")]


def save_results(results, query_file, results_dir):
    """Save results to JSON file."""
    # Create output filename preserving original query filename
    output_file = results_dir / query_file

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to: {output_file}")


def load_playlist_path_index(index_file):
    """Load playlist path index from JSON file."""
    with open(index_file, "r", encoding="utf-8") as f:
        return json.load(f)


def find_playlist_file(playlist_id, playlist_path_index):
    """Find the file path for a given playlist ID."""
    for bucket_key, file_path in playlist_path_index.items():
        min_pid, max_pid = map(int, bucket_key.split("_"))
        if min_pid <= playlist_id <= max_pid:
            return Path(file_path)
    return None


# ============================================================================
# TEXT PREPROCESSING
# ============================================================================
def tokenize(text):
    """Simple word tokenizer and normalizer."""
    return [w.lower() for w in nltk.word_tokenize(text)]
