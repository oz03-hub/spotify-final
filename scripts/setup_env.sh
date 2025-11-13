#!/bin/bash
# Setup script for clean environment

# Remove old environment if it exists
conda deactivate 2>/dev/null || true
conda env remove -n retrieval_env -y 2>/dev/null || true

# Create new conda environment with Python 3.10
conda create -n retrieval_env python=3.10 -y

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate retrieval_env

# Install PyTorch (adjust based on your CUDA version)
# For CUDA 11.8:
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y

# For CUDA 12.1:
# conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# For CPU only:
# conda install pytorch cpuonly -c pytorch -y

# Install other requirements
pip install --upgrade pip
pip install numpy==1.26.4
pip install scipy==1.11.4
pip install transformers>=4.30.0
pip install faiss-cpu==1.7.4
pip install tqdm>=4.65.0
pip install orjson
pip install scikit-learn
pip install implicit
pip install xgboost

echo ""
echo "================================"
echo "Environment setup complete!"
echo "To activate the environment, run:"
echo "conda activate retrieval_env"
echo "================================"