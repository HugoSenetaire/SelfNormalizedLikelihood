# Self-Normalized Likelihood

Code for the paper **"Learning Energy-Based Models by Self-normalising the Likelihood"** (TMLR 2026).

[[Paper]](https://arxiv.org/abs/2503.07021)

## Overview

Training energy-based models (EBMs) with maximum likelihood is challenging due to the intractable normalisation constant. This repo implements **Self-Normalised Log-Likelihood (SNL)**, a novel objective that introduces a single additional learnable parameter representing the normalisation constant. Unlike standard maximum likelihood, SNL can be directly optimised using stochastic gradient methods by sampling from a crude proposal distribution -- no MCMC required.

## Structure

```
Dataset/              # Data loading and preprocessing
Model/                # EBM architectures and proposal distributions
conf_checkerboard/    # Hydra configs for 2D checkerboard experiments
conf_mnist_*/         # Hydra configs for MNIST experiments
conf_uci/             # Hydra configs for UCI datasets
sh_files/             # Shell scripts for running experiments
main_trainer.py       # Main training script (MNIST)
main_trainer_checkerboard.py  # Training script for 2D density estimation
main_trainer_uci.py   # Training script for UCI tabular data
```

## Installation

```bash
git clone --recurse-submodules https://github.com/HugoSenetaire/SelfNormalizedLikelihood.git
cd SelfNormalizedLikelihood
pip install -r requirements.txt  # if available, otherwise install PyTorch + Hydra
```

## Usage

Training is configured via [Hydra](https://hydra.cc/). Example:

```bash
# 2D checkerboard density estimation
python main_trainer_checkerboard.py

# MNIST
python main_trainer.py

# UCI datasets
python main_trainer_uci.py
```

Override config parameters on the command line:

```bash
python main_trainer.py model.learning_rate=1e-4 training.num_epochs=100
```

## Citation

```bibtex
@article{senetaire2026snl,
  title={Learning Energy-Based Models by Self-normalising the Likelihood},
  author={Senetaire, Hugo and Jeha, Paul and Mattei, Pierre-Alexandre and Frellsen, Jes},
  journal={Transactions on Machine Learning Research},
  year={2026}
}
```
