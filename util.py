import orjson
from pathlib import Path
import os


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
