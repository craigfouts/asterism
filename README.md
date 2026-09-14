# Asterism - Experiments with Point Cloud Clustering

[![License](https://img.shields.io/badge/License-Apache_2.0-red.svg)](https://opensource.org/licenses/Apache-2.0) [![Tests](https://github.com/craigfouts/asterism/actions/workflows/asterism.yml/badge.svg?event=push)](https://github.com/craigfouts/asterism/actions/workflows/asterism.yml) [![Python](https://img.shields.io/badge/Python-3.14.6-blue)](https://www.python.org/downloads/release/python-3146/)


**Asterism** is a collection of semantic segmentation models and dimension reduction algorithms applied to point cloud data. Current methods include variations of latent Dirichlet allocation, several neural topic models, and a neural clustering process. It also serves as a test bed for **ATLAS**, a nonparametric neural topic model for discovering an unknown number of spatially-resolved clusters.

## Installation

### MacOS

```bash
git clone https://github.com/craigfouts/asterism.git
cd asterism
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

### Windows

```bash
git clone https://github.com/craigfouts/asterism.git
cd asterism
python -m venv .env
.env\Scripts\activate
pip install -r requirements.txt
```

## Usage

```python
from asterism import ATLAS
from asterism.utils.data import make_dataset
from asterism.utils.plots import show_comparison

data, locs, labels = make_dataset(wiggle=.2, mix=.2, return_tensor=True, seed=0)
topics = ATLAS(seed=0).fit_predict(data, locs, labels)
show_comparison(locs, labels, topics)
```

<img width="100%" alt="image" src="https://github.com/user-attachments/assets/8887bf88-4b21-4e49-ab3d-9f9ed1136f4e" />

Additional examples are provided in ```notebooks/demos.ipynb```.
