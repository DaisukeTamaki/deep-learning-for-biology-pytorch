# Deep Learning for Biology — PyTorch Edition

A community-maintained **PyTorch re-implementation** of the code from [*Deep Learning for Biology*](https://www.oreilly.com/library/view/deep-learning-for/9781098164263/) (Ravarani & Latysheva, O'Reilly, 2025).

The original companion code — the [`dlfb`](https://github.com/deep-learning-for-biology/dlfb) library and the [chapter notebooks](https://github.com/deep-learning-for-biology/notebooks) — is written in **JAX / Flax**. This repository ports those implementations to **PyTorch** so that practitioners more familiar with the PyTorch ecosystem can follow along with the book and build on its ideas.

> **Disclaimer:** This project is not affiliated with, endorsed by, or connected to O'Reilly Media or the book's authors. It is an independent, community-driven effort.

## Contents

| Directory | Description |
|-----------|-------------|
| `dlfb_pytorch/` | PyTorch version of the `dlfb` companion library (preprocessing, models, training, evaluation) |
| `notebooks/` | Executable Jupyter notebooks mirroring the five main project chapters |

### Chapter Notebooks

| Chapter | Topic |
|---------|-------|
| 2 | Learning the Language of Proteins |
| 3 | Learning from DNA Sequences |
| 4 | Understanding Drug–Drug Interactions Using Graphs |
| 5 | Detecting Skin Cancer in Medical Images |
| 6 | Learning Spatial Organization Patterns Within Cells |

## Getting Started

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.0
- A CUDA-capable GPU is recommended but not required

### Installation

```bash
# Clone the repo
git clone https://github.com/DaisukeTamaki/deep-learning-for-biology-pytorch.git
cd deep-learning-for-biology-pytorch

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the library in editable mode with dependencies
pip install -e ".[all]"
```

### Running the Notebooks

You can run the notebooks locally with Jupyter:

```bash
pip install jupyterlab
jupyter lab notebooks/
```

Or upload them to [Google Colab](https://colab.research.google.com/) and select a GPU runtime.

## Project Structure

```
deep-learning-for-biology-pytorch/
├── dlfb_pytorch/
│   ├── __init__.py
│   ├── proteins/       # Protein language models
│   ├── dna/            # DNA sequence models
│   ├── graphs/         # Graph neural networks for drug interactions
│   ├── cancer/         # CNNs / ResNets for medical imaging
│   ├── localization/   # Spatial organization models
│   ├── metrics/        # Precision, recall, F1, etc.
│   └── utils/          # Config, logging, visualization helpers
├── notebooks/
│   ├── chapter_2_proteins.ipynb
│   ├── chapter_3_dna.ipynb
│   ├── chapter_4_graphs.ipynb
│   ├── chapter_5_cancer.ipynb
│   └── chapter_6_localization.ipynb
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Mapping from JAX/Flax to PyTorch

| JAX / Flax | PyTorch Equivalent |
|---|---|
| `flax.linen.Module` | `torch.nn.Module` |
| `optax` optimizers | `torch.optim` |
| `jax.numpy` | `torch` tensor ops |
| `jax.random` | `torch.Generator` / `torch.manual_seed` |
| `orbax` checkpointing | `torch.save` / `torch.load` |
| `dm_pix` image augmentation | `torchvision.transforms` |
| `jraph` (graphs) | `torch_geometric` |

## Citations

This project is based on the code accompanying:

```bibtex
@book{deep_learning_for_biology,
  title     = {Deep Learning for Biology},
  author    = {Ravarani, C. and Latysheva, N.},
  publisher = {O'Reilly Media},
  year      = {2025},
}
```

## License

The code in this repository is licensed under the [Apache 2.0 License](LICENSE).
