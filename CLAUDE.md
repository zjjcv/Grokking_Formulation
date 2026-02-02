# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a theoretical research project investigating the Grokking phenomenon in neural networks - where models suddenly transition from memorization to generalization after prolonged training on modular arithmetic tasks.

## Common Commands

### Training
```bash
# Run Grokking reproduction experiment (x + y mod 97)
python src/grokking_reproduce.py
```

### Visualization
```bash
# Generate training curves from metric.csv
python src/training_plot.py
```

### Deployment
```bash
# Deploy to GitHub (auto-commit and push)
./deploy.sh "commit message"
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Code Architecture

### Core Components

**src/grokking_reproduce.py** - Main training script
- `Config`: Central configuration class (model params, training hyperparams, paths)
- `ModuloDataset`: Generates (x, y) pairs for modular arithmetic, splits 50/50 train/test
- `GrokkingTransformer`: 2-layer Transformer with:
  - Multi-head attention (4 heads, dim 128)
  - Feed-forward networks (dim 512)
  - Learned positional encoding
  - Input format: [x, op_token, y] where op_token = p
- `WarmupCosineScheduler`: Warmup (2000 steps) + cosine decay for 100K total steps
- Training loop saves metrics every 100 steps to CSV and checkpoints

**src/training_plot.py** - Visualization script
- Reads `data/x+y/metric.csv` (columns: step, train_loss, train_acc, test_loss, test_acc)
- Generates dual-plot figure (accuracy + loss) with log-scale x-axis
- Outputs PNG and PDF to `experiments/figures/`

### Data Flow

1. Training generates `data/x+y/metric.csv` and checkpoints to `data/x+y/checkpoints/`
2. Plot script reads metric.csv, outputs to `experiments/figures/training_curves.png`

### Key Configuration

Critical hyperparameters in `Config` class:
- `p = 97`: Modulus for arithmetic
- `weight_decay = 0.1`: High regularization is key for Grokking
- `total_steps = 100000`: Extended training enables delayed generalization
- `save_interval = 100`: Controls metric logging frequency

## Current Status

The x+y (mod 97) experiment has been run but did NOT exhibit Grokking:
- Train accuracy: 100%
- Test accuracy: ~25% (no sudden generalization)

This suggests hyperparameter tuning may be needed (e.g., lower weight decay, different architecture).
