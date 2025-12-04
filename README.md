# CS646 Spotify Playlist Generator

## Set up
Clone this repository.

Download the dataset from: [https://drive.google.com/file/d/1O9euNDgvpkyG0sa7oYya4E5fnPjsqYYS/view?usp=sharing](https://drive.google.com/file/d/1NOiqWRXLmNz58Wefq8bdzQlNKTrStLzN/view?usp=sharing), place the zip at the root of the repository. Unzip the small dataset, this is the smaller version with 10,000 playlists. It should be unzipped into dataset directory. To confirm verify `dataset/train/`, `dataset/test/`, `dataset/tracks_index.json` exists after unzipping. Under `dataset`, the `playlist_metadata.json` file must be located too, if unzipping erased or corrupted it, please download again and place it there.

You need a fresh python environment. 
1. Run `./scripts/setup_env.sh`. (If it does not work, please resolve the error with your packager, the libraries listed there must be installed). If any further script throws missing library error, please activate the environment and install it. You can activate the environment with `conda activate retrieval_env`.
2. With the environment activated run `python build_inverted_index.py` to verify it is working.

# Running the baselines
**For all baselines**, we recommend a CPU with multiple cores, and around 32GB of memory. Unity research cluster is more than capable of running all experiments, personal laptops might be slightly slower and toasty. For fine-tuning the dense retriever, you should use a GPU, we provide slurm scripts to tune on Unity.

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

or 

```
sbatch make_rerank_data.sh
```

This will create training data and re-ranked test/validation data.

### Training LTR Model
```
python ltr.py
```

## Hybrid Retrieval
### Fine-tuning DPR
You can use the slurm script, but must change the checkpoint location:
```
scbatch scripts/fine_tune.sh
```
Or you can run the python command itself.

### Training and inference
Make sure the model directory is pointed towards the checkpoint location you used for DPR fine-tuning.
```
python hybrid_ret.py --retrain --model_dir /path/to/a/hpc/space/
```

## Transfer Based Initialization
The transfer based initialization adds an extra step to the original zero-shot approach. We get the top $$k_p$$ pre-existing playlists closest to the provided title, using the songs in those playlists alongside the original title to generate the top $$k_f$$ final songs

### Create Narrow JSON
```
python narrow_json.py
```
Saves a smaller file that contains only the information needed for the transfer based initialization

### WMF Playlists
```
python wmf_playlists.py
```

Saves 3 seperate results, see output for file locations:
1. Top $$k_P$$ playlists found based on inputted playlist title
2. Top $$k_f$$ final songs with averaged song embeddings
3. Top $$k_f$$ final songs with a weighted average over song embeddings 


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
