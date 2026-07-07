# street-pattern-classifier

---

[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

Built with:

![numpy](https://img.shields.io/badge/NumPy-013243.svg?style={0}&logo=NumPy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458.svg?style={0}&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikitlearn-F7931E.svg?style={0}&logo=scikit-learn&logoColor=white)
![scipy](https://img.shields.io/badge/SciPy-8CAAE6.svg?style={0}&logo=SciPy&logoColor=white)
![tqdm](https://img.shields.io/badge/tqdm-FFC107.svg?style={0}&logo=tqdm&logoColor=black)

---

## Table of Contents

- [Overview](#overview)
- [Core Features](#core-features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Overview

street-pattern-classifier is a Python project for classifying urban street blocks from graph and morphology-derived features. It is aimed at developers and researchers working in urban morphology, geospatial analysis, and graph-based machine learning. The repository is notebook-driven and supports a workflow that builds block graphs, prepares datasets, runs classification, and visualizes results; for a runnable path through that workflow, start with Getting Started and the usage notebook.

---

## Core Features

- Build street-block graphs from urban road networks and block polygons, giving developers a geometry-aware representation of city form for downstream analysis.
- Derive block-level morphological features such as area, circuity, concavity, elongation, and rectangularity, which supports pattern classification from measurable urban shape properties.
- Convert prepared block graphs into graph-structured ML data with node and edge features, enabling use with PyTorch Geometric-style models.
- Classify street-pattern subgraphs into the project’s supported label set, so developers can predict morphology classes for new city areas.
- Visualize predicted subgraphs, polygons, and class-wise features, making it easier to inspect results and compare street-pattern types.

---

## Installation

Install street-pattern-classifier using one of the following methods:

**Build from source:**

1. Clone the street-pattern-classifier repository:
```sh
git clone https://github.com/GeorgeKontsevik/street-pattern-classifier
```

2. Navigate to the project directory:
```sh
cd street-pattern-classifier
```

3. Install the project dependencies:

```sh
pip install -r requirements.txt
```

---

## Getting Started

**Prerequisites**

- Python environment with the project dependencies installed.
- A city road graph available through `osmnx`.
- Access to the pretrained model file `best_model.pth` from the Hugging Face repository `nochka/street-pattern-classifier`.

**Quick start**

1. Open `usage.ipynb`.
2. Download and project a city graph:

```python
   import osmnx as ox

   place = "Paris, France"

   G = ox.graph_from_place(place, network_type="drive", simplify=True)
   G = ox.project_graph(G)
```

3. Download the model file into `./models`:

```python
   from huggingface_hub import hf_hub_download
   import torch

   model_path = hf_hub_download(
       repo_id="nochka/street-pattern-classifier",
       filename="best_model.pth",
       local_dir="./models"
   )
```

4. Split the graph into subgraphs and build the dataset:

```python
   from splits import split_graph
   from block_dataset import BlockDataset

   subgraphs = split_graph(G, grid_step=2000)
   dataset = BlockDataset(subgraphs)
```

5. Run classification:

```python
   from classification import classify_blocks

   predictions_blocks, probabilities_blocks = classify_blocks(
       dataset,
       model_path=model_path,
       device='cuda'
   )
```

6. Plot the results if needed:

```python
   from plots import plot_all_subgraphs
   from model import class_names

   plot_all_subgraphs(subgraphs, predictions_blocks, class_names, (20,20))
```

---

## Architecture

This repository is organized as a notebook-driven Python workflow for classifying street-pattern types from urban block geometry and road networks.

- A street graph is loaded with `osmnx`, then split into smaller subgraphs or grid cells (`splits.py`) for per-block processing.
- Each subgraph is converted into street blocks by buffering road geometries and subtracting them from the enclosing polygon (`block_graph.py`).
- The resulting block polygons are turned into a graph where nodes represent blocks and edges connect neighboring blocks; node and edge attributes are then derived from geometry and road context (`block_graph.py`, `classification.py`).
- `BlockDataset` prepares these block graphs for PyTorch Geometric, normalizing node features and exposing graph tensors for inference (`block_dataset.py`).
- `classification.py` loads a trained model and runs block-level prediction to produce class labels and probabilities.
- `model.py` defines the available model configurations and the class label names used by the workflow, while `plots.py` is used for visualizing subgraphs, polygons, and feature patterns.

The provided usage notebook shows the intended end-to-end flow: download a city graph, fetch a model checkpoint from Hugging Face, split the graph, build the dataset, classify blocks, and visualize the results.

---

## Documentation

A detailed street-pattern-classifier description is available [here](https://github.com/GeorgeKontsevik/street-pattern-classifier/tree/main/docs).

---

## Contributing

- **[Report Issues](https://github.com/GeorgeKontsevik/street-pattern-classifier/issues)**: Submit bugs found or log feature requests for the project.

- **[Submit Pull Requests](https://github.com/GeorgeKontsevik/street-pattern-classifier/tree/main/CONTRIBUTING.md)**: To learn more about making a contribution to street-pattern-classifier.

---

## Citation

If you use this software, please cite it as below.

### APA format:

    GeorgeKontsevik (2026). street-pattern-classifier repository [Computer software]. https://github.com/GeorgeKontsevik/street-pattern-classifier

### BibTeX format:

    @misc{street-pattern-classifier,

        author = {GeorgeKontsevik},

        title = {street-pattern-classifier repository},

        year = {2026},

        publisher = {github.com},

        journal = {github.com repository},

        howpublished = {\url{https://github.com/GeorgeKontsevik/street-pattern-classifier}},

        url = {https://github.com/GeorgeKontsevik/street-pattern-classifier}

    }

---