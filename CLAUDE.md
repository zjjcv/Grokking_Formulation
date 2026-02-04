# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive research project investigating the **Grokking phenomenon** through systematic frequency domain analysis. The project includes:

1. **Grokking reproduction** - Training Transformer models on four modular arithmetic operations (x+y, x-y, x*y, x÷y mod 97)
2. **Spectral analysis** - 1D and 2D DFT analysis across 1000 training checkpoints
3. **Multiple analysis dimensions** - Sparsity, phase, circuit competition, spectral attribution, effective rank, flatness (SLT), group representation
4. **Unified analysis framework** - Consistent pipeline for analyzing all four operations

**Key Finding**: Grokking emerges from convergence of sparsification, circuit reorganization, and spectral restructuring---not simple frequency alignment.

## Dataset and Task Details

### Modulo Arithmetic Task
- **Domain**: integers modulo 97 (prime number)
- **Input format**: [x, operation_token, y] where operation_token = p (97)
- **Output**: single integer in [0, 96]
- **Train/test split**: 50/50 (4705 train, 4704 test samples)
- **Random seed**: 42 (fixed for reproducibility)

### Checkpoint Format
Each checkpoint file (`checkpoint_step_N.pt`) contains:
```python
{
    'model_state_dict': {
        'embedding.weight': (98, 128),
        'pos_encoding': (1, 3, 128),
        'blocks.0.attn.W_Q', 'W_K', 'W_V', 'W_O': (4, 32, 128),
        'blocks.0.ffn.0.weight', 'ffn.2.weight': (512, 128), (128, 512),
        'blocks.1.*': (same as block 0),
        'ln_1.weight', 'ln_1.bias', 'ln_2.weight', 'ln_2.bias': (128,),
        'fc_out.weight', 'fc_out.bias': (97, 128), (97,)
    },
    'train_loss': float,
    'train_acc': float,
    'test_loss': float,
    'test_acc': float,
    'step': int
}
```

## Common Commands

### Training (Four Operations)
```bash
# 使用统一训练脚本（推荐）
python scripts/training/train.py --operation add   # x + y mod 97
python scripts/training/train.py --operation sub   # x - y mod 97
python scripts/training/train.py --operation mul   # x * y mod 97
python scripts/training/train.py --operation div   # x ÷ y mod 97

# 或者使用独立脚本
python scripts/training/train_addition.py
python scripts/training/train_subtraction.py
python scripts/training/train_multiplication.py
python scripts/training/train_division.py
```

### Unified Analysis Pipeline (Recommended)
```bash
# Run all analyses for all operations (data collection + plotting)
python scripts/run_pipeline.py --all

# Run for specific operation only
python scripts/run_pipeline.py --operation x+y

# Skip data collection, only generate plots
python scripts/run_pipeline.py --all --skip-analysis
```

### Individual Data Collection Analyses
```bash
# 2D DFT analyses (unified framework)
python scripts/analysis/2d_dft/gini_ipr.py --operation x+y
python scripts/analysis/2d_dft/f_alignment.py --operation x+y
python scripts/analysis/2d_dft/phase_coherence.py --operation x+y
python scripts/analysis/2d_dft/fourier_projection.py --operation x+y
python scripts/analysis/2d_dft/spectral_attribution.py --operation x+y
python scripts/analysis/2d_dft/circuit_competition.py --operation x+y
python scripts/analysis/2d_dft/qk_circuit.py --operation x+y

# Additional analyses (all operations)
python scripts/analysis/effective_rank.py --all
python scripts/analysis/flatness_slt.py --all
python scripts/analysis/group_representation.py --all
```

### Visualization
```bash
# Unified plotting scripts (all support --operation argument)
python scripts/plotting/training_curves.py --operation x+y
python scripts/plotting/2d_dft/gini_ipr.py --operation x+y
python scripts/plotting/2d_dft/f_alignment.py --operation x+y
python scripts/plotting/2d_dft/fourier_projection.py --operation x+y
python scripts/plotting/2d_dft/circuit_competition.py --operation x+y
python scripts/plotting/2d_dft/spectral_attribution.py --operation x+y
python scripts/plotting/2d_dft/qk_circuit.py --operation x+y
python scripts/plotting/2d_dft/qk_spectrum.py --operation x+y

# Plot for all operations
for op in x+y x-y x*y x_div_y; do
    python scripts/plotting/training_curves.py --operation $op
done
```

### Legacy Individual Scripts (Archived)

> **Note**: These scripts have been moved to `archive/legacy/` for reference. Use the unified scripts in `scripts/` for new work.

```bash
# 1D DFT analyses (archived)
python archive/legacy/individual_scripts/f_alginment.py
python archive/legacy/individual_scripts/psd.py
python archive/legacy/individual_scripts/gini_ipr.py
python archive/legacy/individual_scripts/Phase_Coherence.py
python archive/legacy/individual_scripts/Fourier_Projection.py

# Original 2D DFT analyses (archived)
python archive/legacy/2D_DFT/f_alginment.py
python archive/legacy/2D_DFT/psd.py
python archive/legacy/2D_DFT/gini_ipr.py
python archive/legacy/2D_DFT/Phase_Coherence.py
python archive/legacy/2D_DFT/Fourier_Projection.py
python archive/legacy/2D_DFT/Phase_Intervention.py
python archive/legacy/2D_DFT/Constellation_Evolution.py
python archive/legacy/2D_DFT/Spectral_Attribution.py

# Circuit analyses (archived)
python archive/legacy/individual_scripts/circuit_Competition.py
python archive/legacy/individual_scripts/intrinsic_dimension.py
python archive/legacy/individual_scripts/qk_circut.py
```

### Dependencies
```bash
pip install -r requirements.txt  # torch>=2.0.0, numpy>=1.24.0, matplotlib
```

### Deployment
```bash
./deploy.sh "commit message"  # Deploy to GitHub (git add + commit + push)
```

## Code Architecture

### Data Flow Pipeline

```
Training → Checkpoints → Data Collection → Visualization → Paper
   ↓           ↓              ↓                ↓            ↓
grokking_  1000 ×      18 × CSV       48 ×        LaTeX
reproduce  .pt files   (raw data)    PNG/PDF      → PDF
.py                                   figures
```

### Directory Structure

```
Grokking_Formulation/
├── scripts/
│   ├── training/                  # Training scripts
│   │   ├── train.py               # Unified training script (recommended)
│   │   ├── train_addition.py      # x + y mod 97
│   │   ├── train_subtraction.py   # x - y mod 97
│   │   ├── train_multiplication.py # x * y mod 97
│   │   └── train_division.py      # x ÷ y mod 97
│   │
│   ├── analysis/                  # Data collection analysis
│   │   ├── 2d_dft/                # 2D DFT analyses
│   │   │   ├── gini_ipr.py
│   │   │   ├── f_alignment.py
│   │   │   ├── phase_coherence.py
│   │   │   ├── fourier_projection.py
│   │   │   ├── spectral_attribution.py
│   │   │   ├── circuit_competition.py
│   │   │   ├── qk_circuit.py
│   │   │   └── qk_circuit_full.py
│   │   ├── effective_rank.py
│   │   ├── flatness_slt.py
│   │   ├── group_representation.py
│   │   └── circuit_competition_full.py
│   │
│   ├── plotting/                  # Visualization scripts
│   │   ├── 2d_dft/
│   │   ├── training_curves.py
│   │   ├── effective_rank.py
│   │   ├── flatness_slt.py
│   │   ├── group_representation.py
│   │   └── circuit_competition_full.py
│   │
│   ├── lib/                       # Shared library
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration & styling
│   │   └── utils.py               # Utility functions
│   │
│   ├── run_pipeline.py            # Main pipeline controller
│   └── README.md                  # Scripts documentation
│
├── archive/legacy/                # Archived old scripts
│   ├── 2D_DFT/
│   ├── plot_scripts/
│   └── individual_scripts/
│
├── data/
│   ├── x+y/                       # Addition operation data
│   │   ├── checkpoints/           # 1000 checkpoint files
│   │   └── *.csv                  # Analysis results
│   ├── x-y/                       # Subtraction (same structure)
│   ├── x*y/                       # Multiplication (same structure)
│   └── x_div_y/                   # Division (same structure)
│
├── experiments/
│   ├── figures/                   # Generated plots
│   │   ├── x+y/, x-y/, x*y/, x_div_y/
│   └── logs/
│
├── writing/
│   ├── paper/                     # LaTeX paper source
│   └── notes/                     # Research notes
│
├── CLAUDE.md
├── README.md
└── requirements.txt
```

### Core Analysis Scripts

Each analysis script follows the same pattern:

```python
# Data collection (scripts/analysis/2d_dft/XXX.py)
1. Load checkpoints from data/{operation}/checkpoints/
2. Extract weights (W_E, W_U, attention matrices, etc.)
3. Compute analysis metrics (DFT, IPR, projections, etc.)
4. Save to data/{operation}/XXX.csv

# Visualization (scripts/plotting/2d_dft/XXX.py)
1. Load data/{operation}/XXX.csv
2. Create matplotlib figures (multi-panel, log-scale, annotations)
3. Save to experiments/figures/{operation}/XXX.png and XXX.pdf
```

**Checkpoints are loaded with `torch.load(checkpoint_path, map_location='cpu')` for CPU compatibility.**

**Common weights extracted:**
- `embedding.weight` (W_E): Token embeddings, shape (98, 128)
- `fc_out.weight` (W_U): Output unembedding, shape (97, 128)
- `blocks.{i}.attn.{W_Q,W_K,W_V,W_O}`: Attention weights for each head

### Unified Framework (`scripts/lib/config.py`)

The project uses a unified configuration system for consistency across all four operations:

- **OPERATIONS dict**: Defines data directories, colors, and symbols for x+y, x-y, x*y, x_div_y
- **COLORS dict**: Unified color scheme for all metrics (train_acc, test_acc, gini, alignment, etc.)
- **FONTS, LINES, MARKERS**: Consistent styling across all plots
- **Helper functions**: `setup_style()`, `save_figure()`, `add_grokking_region()`, `add_crossover_line()`

All unified scripts accept `--operation` parameter and use `lib.config` for paths and styling.

### Key Analysis Dimensions

| Analysis | Metric | Key Finding | CSV File |
|----------|--------|-------------|-----------|
| **Sparsity** | IPR, Gini | IPR ↑ 5× at 20K (precedes Grokking) | gini_ip_2d.csv |
| **1D Alignment** | Cosine sim | Increases 0.62→0.89 | f_alginment.csv |
| **2D Alignment** | Cosine sim | Decreases 0.78→0.68 (contradicts 1D) | f_alginment_2d.csv |
| **Fourier Sparsity** | L1/L2, Gini | W_U Fourier ↓ 30% | fourier_projection_2d.csv |
| **Phase Coherence** | R² (linearity) | Remains ~0.016 (no linear structure) | phase_coherence_2d.csv |
| **Phase Intervention** | Accuracy | Ideal phases destroy 100%→1% | phase_intervention.csv |
| **Circuit Competition** | Projection ratio | Crossover at 22K (precedes Grokking) | circuit_Competition.csv |
| **Spectral Attribution** | Frequency energy | Stable distribution, no magic frequency | spectral_attribution.csv |
| **Effective Rank** | exp(H(W)) | Measures rank reduction via SVD | effective_rank.csv |
| **Flatness (SLT)** | λ_eff | Learning coefficient via SLT | flatness_slt.csv |
| **Group Representation** | εR, δorth | Ring structure in embeddings | group_representation.csv |
| **Constellation** | Complex plane | No circular organization | constellation_data.csv |
| **Intrinsic Dimension** | Dimension | W_E decreases, W_U U-shaped | intrinsic_dimension.csv |

### Model Architecture

All four operations use the same architecture for consistency:

```python
class GrokkingTransformer(nn.Module):
    - vocab_size = p + 1 = 98
    - embedding: nn.Embedding(98, 128)
    - pos_encoding: nn.Parameter(torch.randn(1, 3, 128))
    - 2 × TransformerBlock (d_model=128, n_heads=4, d_ff=512, dropout=0.1)
    - MultiHeadAttention with causal=False (bidirectional)
    - FFN: Linear(128→512) → ReLU → Linear(512→128)
    - output: nn.Linear(128, 97)
```

**Critical hyperparameters (identical for all operations):**
- `p = 97` (modulus)
- `weight_decay = 0.005` (enables Grokking)
- `total_steps = 100000`
- `save_interval = 100` (creates 1000 checkpoints)
- `train/test split = 50/50` (fixed seed 42)

**Training requirements:**
- GPU recommended but not required (code auto-detects CUDA availability)
- Training time: ~2-4 hours per operation on GPU, ~20-40 hours on CPU
- GPU memory: <2GB VRAM sufficient (model is small)

**Operation mappings:**
- Addition: `operation = "add"`, label = `(x + y) % p`, training script: `grokking_reproduce.py`
- Subtraction: `operation = "sub"`, label = `(x - y) % p`, training script: `grokking_reproduce_x_minus_y.py`
- Multiplication: `operation = "mul"`, label = `(x * y) % p`, training script: `grokking_reproduce_x_mul_y.py`
- Division: `operation = "div"`, label = `(x * y^{-1}) % p`, training script: `grokking_reproduce_x_div_y.py`

**Note on division**: Uses modular inverse for y (y⁻¹ mod p). Since p=97 is prime, all y in [1,96] have valid inverses. y=0 is excluded or handled specially.

## Current Status

**Experiments completed:**
- ✅ Grokking reproduction for all 4 operations (100K steps, 1000 checkpoints each)
- ✅ All 1D DFT analyses (5 types)
- ✅ All 2D DFT analyses (8 types) - both original and unified versions
- ✅ Circuit competition analysis
- ✅ Intrinsic dimension analysis
- ✅ QK circuit spectral analysis
- ✅ Effective rank analysis (new)
- ✅ Flatness analysis via SLT (new)
- ✅ Group representation analysis (new)
- ✅ Unified framework implemented for all analyses
- ✅ All visualizations generated (48+ figures per operation)
- ✅ Complete paper written

**Paper ready for submission** (needs author info, acknowledgments)

## Operation Differences

While all four operations use identical model architecture and training hyperparameters, they exhibit different Grokking behaviors:

| Operation | Grokking Onset | Difficulty | Notes |
|-----------|---------------|------------|-------|
| x+y       | ~30K steps    | Easiest    | Most stable convergence |
| x-y       | ~35K steps    | Easy       | Similar to addition |
| x*y       | ~70K steps    | Hard       | Requires more exploration |
| x÷y       | ~80K steps    | Hardest    | Modular inverse adds complexity |

**Key insight**: The computational difficulty correlates with Grokking onset time. Multiplication and division involve more complex group structure (multiplicative group mod 97) compared to additive operations.

## Key Findings Summary

1. **Sparsification precedes generalization**: IPR peaks at 20K steps, 10K before test accuracy improves
2. **Circuit crossover drives transition**: Algorithm subspace overtakes memorization at 22K steps
3. **2D spectral structure is complex**: 2D alignment decreases (0.78→0.68), contradicting 1D results
4. **Phase structure is non-linear**: Ideal phase intervention destroys performance (100%→1%)
5. **No magic frequency**: Spectral attribution shows stable distribution across bands

## Important Notes

- **All checkpoint files are large**: Each .pt file is several MB, ~4000 files total (1000 × 4 operations)
- **Use unified scripts**: For new analyses, follow the pattern in `2D_DFT_unified/` and `plot_unified/` rather than the original scripts
- **Import analysis_config**: All unified scripts should `from analysis_config import OPERATIONS, get_checkpoint_dir` for consistency
- **Operation-specific paths**: All unified scripts use `--operation` argument to select which operation to analyze
- **1D vs 2D DFT contradiction**: This is a critical finding that 1D analysis misses bidirectional spectral patterns
- **Phase intervention negative result**: The fact that "ideal" phases destroy performance is itself an important finding
- **Temporal resolution**: 100-step intervals enable precise correlation of metrics with Grokking onset
- **Colors by operation**: Each operation has assigned primary/secondary colors (see `analysis_config.py`)
- **No test suite**: This project has no unit tests. Analyses are validated by comparing results across operations
- **Hardcoded paths**: Original `2D_DFT/` and `plot/` scripts contain hardcoded paths. Use `2D_DFT_unified/` and `plot_unified/` for flexibility

## New Analyses (Beyond Original Paper)

1. **Effective Rank** (`effective_rank.py`): Computes spectral entropy and effective rank via SVD for all weight matrices
2. **Flatness (SLT)** (`flatness_slt.py`): Estimates learning coefficient λ using Singular Learning Theory
3. **Group Representation** (`group_representation.py`): Analyzes ring structure in embeddings using Orthogonal Procrustes
4. **QK Circuit** (`qk_circut.py`): Spectral analysis of Query-Key attention circuits

## Adding New Analyses

To add a new analysis following the unified framework:

**1. Data Collection Script** (`scripts/analysis/2d_dft/new_analysis.py`):
```python
#!/usr/bin/env python3
import os, sys, argparse, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.config import OPERATIONS
from lib.utils import get_checkpoint_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--operation', default='x+y', choices=list(OPERATIONS.keys()))
    args = parser.parse_args()

    checkpoint_dir = get_checkpoint_dir(args.operation)
    output_file = f"{OPERATIONS[args.operation]['data_dir']}/new_analysis.csv"

    # Load checkpoints, compute metrics, save to CSV
    ...

if __name__ == "__main__":
    main()
```

**2. Visualization Script** (`scripts/plotting/2d_dft/new_analysis.py`):
```python
#!/usr/bin/env python3
import os, sys, argparse, matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.config import OPERATIONS, setup_style, save_figure, COLORS
from lib.utils import get_figures_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--operation', default='x+y', choices=list(OPERATIONS.keys()))
    args = parser.parse_args()

    setup_style()
    data_file = f"{OPERATIONS[args.operation]['data_dir']}/new_analysis.csv"
    output_dir = get_figures_dir(args.operation)

    # Load CSV, create plots, save figures
    ...

if __name__ == "__main__":
    main()
```

**3. Register in Pipeline** (`scripts/run_pipeline.py`):
Add to `ANALYSIS_SCRIPTS` or `PLOT_SCRIPTS` dicts as needed.

## Plotting Conventions

### Standard Elements
- **X-axis**: Training steps (log scale), range [100, 100000]
- **Grokking region**: Shaded area starting at step 30000 (yellow, alpha=0.1)
- **Crossover line**: Vertical dashed line at circuit crossover point (~22000)
- **Dual Y-axis**: Used for correlation plots (e.g., accuracy + metric)

### Figure Format
- DPI: 300 for PNG, vector for PDF
- Font: serif (DejaVu Serif, Times New Roman)
- Grid: enabled with alpha=0.3
- Legend: top-left or upper-left position

### Color Coding by Metric
- `train_acc`: Blue (#3498db)
- `test_acc`: Red (#e74c3c)
- `gini`: Green (#27ae60)
- `ipr`: Purple (#8e44ad)
- `alignment`: Orange (#f39c12)
- `memory`: Dark red (#c0392b)
- `algorithm`: Dark blue (#2980b9)

## Data Format Reference

### checkpoint Loading Pattern
```python
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint['model_state_dict']
embedding = state_dict['embedding.weight']  # (98, 128)
W_U = state_dict['fc_out.weight']  # (97, 128)
```

### CSV Files (Analysis Results)
All analysis CSV files share a common structure:
```csv
step,train_loss,train_acc,test_loss,test_acc,<metric1>,<metric2>,...
0,4.571,0.023,4.572,0.023,0.123,45.6
100,4.123,0.089,4.125,0.087,0.145,43.2
...
100000,0.001,1.000,0.015,0.995,0.987,12.3
```

### Checkpoint Naming Convention
- Pattern: `checkpoint_step_<N>.pt` where N ∈ {0, 100, 200, ..., 100000}
- Total: 1001 checkpoints per operation
- Saved every `save_interval=100` steps

### Metric File (metric.csv)
Generated during training, contains:
```csv
step,train_loss,train_acc,test_loss,test_acc
0,4.571,0.023,4.572,0.023
100,4.123,0.089,4.125,0.087
...
```
