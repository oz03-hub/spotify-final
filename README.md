# CS646 Spotify Playlist Generator

## Set up
Clone this repository.


Download the dataset from: https://drive.google.com/file/d/19PdE-wpWuSUIId6nDhvwEhT6sjfIilXD/view?usp=sharing, place the zip at the root of the repository. Unzip the small dataset, this is the smaller version with 10,000 playlists. It should be unzipped into dataset directory. To confirm verify `dataset/train/`, `dataset/test/`, `dataset/tracks_index.json` exists after unzipping.

You need a fresh python environment. 
1. Create a python environment `python -m venv venv`.
2. Activate the environment `source venv/bin/active`.
3. Install packages `pip install -r requirements.txt`.

If you use conda, the first two steps will be different:
1. Create a conda environment `conda create --name ENV_NAME`.
2. Activate the environment `conda activate ENV_NAME`.
3. Install packages `conda install --file requirements.txt`.

## Running the baselines
Each `*_baseline.py` will output it's ranked results into `dataset/results/*_baseline/`.

### Random Baseline
Random baselines randomly sample songs from the corpus. **With your python environment activated**, run the following:

```
python random_baseline.py
```

Notice in `dataset/results/random_baseline/`, files will be added, or existing ones will be overridden.

### Dirichlet-Smoothing LM
```
python lmir_baseline.py
```

All configurations for smoothing and datapath etc, are made in the python file itself.

This will also save generated paylists under `dataset/results/lmir_baseline/`.

### Matrix Factorization on Song-Playlist Entities
This approach follows the cold-start example used in vl6 method in the survey paper.
```
python mf_baseline.py
```

## Viewing Evaluation Measures

```
python view_measures.py
```

This will display the mean performance of the baseline. To change the baseline being evaluated, you need to change `baseline_name` variable in the code.

# ToDo
[X] Add cli arguments for each baseline/measure, so we don't have to modify code to run different settings
[ ] Add closest playlist name retriever, probably add to `mf_baseline.py`
[X] Switch LMIR to inverted index implementation.
[ ] Add WMF implementation
[ ] Start collecting training data for second stage
