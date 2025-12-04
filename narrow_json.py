import json
import os
from pathlib import Path
from tqdm import tqdm


def condense_json_files(input_folder, output_file):
    """
    Condense all JSON files in a folder into a single JSON file with only pid and track_uri.
    
    Args:
        input_folder: Path to folder containing JSON files
        output_file: Path to output condensed JSON file
    """
    input_path = Path(input_folder)
    json_files = sorted(input_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_folder}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    condensed_data = {}
    
    # Process each JSON file
    for json_file in tqdm(json_files, desc="Processing files"):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract playlists
        playlists = data.get("playlists", [])
        
        for playlist in playlists:
            pid = playlist.get("pid")
            tracks = playlist.get("tracks", [])
            
            # Extract track URIs
            track_uris = [track.get("track_uri") for track in tracks if track.get("track_uri")]
            
            # Add to condensed data
            condensed_data[pid] = track_uris
    
    # Save condensed data
    print(f"\nSaving condensed data to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(condensed_data, f, indent=2)


if __name__ == "__main__":
    condense_json_files("dataset/train", "dataset/narrow/train_narrow.json")