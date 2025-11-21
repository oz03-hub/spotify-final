# CS646 Spotify Playlist Generator

## Set up
Clone this repository.


Download the dataset from: https://drive.google.com/file/d/1O9euNDgvpkyG0sa7oYya4E5fnPjsqYYS/view?usp=sharing, place the zip at the root of the repository. Unzip the small dataset, this is the smaller version with 10,000 playlists. It should be unzipped into dataset directory. To confirm verify `dataset/train/`, `dataset/test/`, `dataset/tracks_index.json` exists after unzipping.

You need a fresh python environment. 
1. Run `./scripts/setup_env.sh`. (If it does not work, please resolve the error with your packager, the libraries listed there must be installed).
2. With the environment activated run `python build_inverted_index.py` to verify it is working.

# Running the baselines
**For all baselines**, we recommend a CPU with multiple cores, and around 32GB of memory. Unity research cluster is more than capable of running all experiments, personal laptops might be slightly slower and warm. For fine-tuning the dense retriever, you should use a GPU, we provide slurm scripts to tune on Unity.

**For Unity users**, the SLURM files are located in `scripts`, you can invoke each by `sbatch file run_*.sh`

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

This will also save generated paylists under `dataset/results/lmir_baseline/`. Later baselines follow the same format.

### BM25
```
python bm25_baseline.py
```

### SVD Matrix Factorization on Song-Playlist Entities
This approach follows the cold-start example used in vl6 method in the survey paper.
```
python svd_baseline.py
```

### Weighted Matrix Factorization on Song-Playlist Entities
```
python wmf_baseline.py
```

# Main Approaches
Please activate your environment as you did in baselines.

## Learning to Rank
### Creating training data
```
python fast_rerank_dataset.py
```

This will create training data and re-ranked test/validation data.

### Training LTR Model
```
python ltr.py
```

## Hybrid Retrieval
### Training and inference
```
python hybrid_ret.py --retrain --model_dir /path/to/a/hpc/space/
```

## Viewing Evaluation Measures

```
python view_measures.py --baseline lmir_baseline
```

To change the baseline viewed, just change the argument for the approach or baseline you ran.

## Inference

You should run the Hybrid Retriever Approach to train and build the index first.

You should use a GPU, best way is to run an interactive Unity job with 1 GPU, anything with >16GB should be fine.

```
python inference.py
```
