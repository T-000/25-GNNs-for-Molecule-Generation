# Molecule Generation with Graph Diffusion (GNN-based)

*A hands-on tutorial notebook that walks through the core ideas behind **DiGress** (discrete diffusion for graphs) and implements a lightweight version with PyTorch Geometric, trained on a subset of **QM9**. Adapted from a TeachOpenCADD template.*

**Author:** Jilixin Tang (Saarland University, 2024)

---

## What this notebook covers

- Why diffusion for **molecular graphs** and why **discrete** noise (vs. Gaussian on continuous features).
- Forward (noising) and reverse (denoising) Markov processes for graphs.
- A **cosine schedule** for discrete diffusion timesteps.
- A minimal **Graph Transformer** that predicts node and edge classes during reverse steps.
- Training on **QM9** (small subset) and simple graph **generation** from noise.

---

## Requirements

- Python 3.9+
- `numpy`
- `torch`
- `torch_geometric`


## How to Run

```bash
# clone the repo
git clone https://github.com/T-000/25-GNNs-for-Molecule-Generation.git
cd 25-GNNs-for-Molecule-Generation

# install dependencies
pip install -r requirements.txt

# run the notebook
jupyter notebook DiGress.ipynb
```

## Note on PyTorch Geometric

PyTorch Geometric requires **version-specific companion packages**  
(`torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`).  

If installation fails with wheel errors, follow the official guide here:  
 [PyG Installation Instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

---

## Results

- Configured for **educational runs**, not SOTA.  
- Default: 20 epochs, 10 diffusion steps, ~1000 molecules.  
- Produces decreasing loss and valid graph structures.  
- For better results: increase dataset size, model width, epochs, and diffusion steps (~500 in the paper).  

---

## References

- Vignac et al., *DiGress: Discrete Denoising Diffusion for Graph Generation*.  
- QM9 dataset via PyTorch Geometric.  
- TeachOpenCADD tutorial templates.  


