#!/bin/bash

# Check if the environment exists and remove it if it does
if conda env list | grep -q 'cvae'; then
  echo "Environment 'cvae' exists. Removing it..."
  conda remove --name cvae --all
fi

# Create a new environment with the essential packages
conda create -n cvae python=3.9 pytorch torchvision torchaudio -c pytorch -c nvidia

# Activate the environment
conda activate cvae

# Install additional packages
conda install -y -n cvae pandas scikit-learn -c conda-forge
conda install -y -n cvae pytorch-cuda=11.7 -c pytorch -c nvidia

# Install pip packages
pip install biobricks
pip install dvc
pip install tqdm
pip install rdkit
pip install matplotlib
pip install dask
pip install "dask[distributed]"
pip install selfies==2.1.1
pip install pyspark
pip install torch

# add ./ to the conda python path so that ./cvae can be imported
conda env config vars set PYTHONPATH="./:$PYTHONPATH" -n cvae
