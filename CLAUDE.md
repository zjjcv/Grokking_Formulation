# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive research project investigating the **Grokking phenomenon** through systematic frequency domain analysis. The project includes:

1. **Grokking reproduction** - Training Transformer models on modular arithmetic (x + y mod 97)
2. **Spectral analysis** - 1D and 2D DFT analysis across 1000 training checkpoints
3. **Multiple analysis dimensions** - Sparsity, phase, circuit competition, spectral attribution, etc.
4. **Complete paper** - arXiv-format paper with figures and appendix

**Key Finding**: Grokking emerges from convergence of sparsification, circuit reorganization, and spectral restructuring---not simple frequency alignment.

## Common Commands

### Training
```bash
# Run Grokking reproduction (x + y mod 97, 100K steps)
python src/grokking_reproduce.py
```

### Data Collection (all analyses use checkpoints from training)
```bash
# 1D DFT analyses
python src/f_alginment.py           # Frequency alignment (1D)
python src/psd.py                   # Power spectral density
python src/gini_ipr.py             # Gini coefficient + IPR
python src/Phase_Coherence.py      # Phase linearity
python src/Fourier_Projection.py   # Fourier vs spatial sparsity

# 2D DFT analyses
python src/2D_DFT/f_alginment.py           # Frequency alignment (2D)
python src/2D_DFT/psd.py                   # 2D Power spectral density
python src/2D_DFT/gini_ipr.py             # 2D Gini + IPR
python src/2D_DFT/Phase_Coherence.py      # 2D Phase analysis
python src/2D_DFT/Fourier_Projection.py   # 2D Fourier projection
python src/2D_DFT/Phase_Intervention.py   # Phase intervention experiment
python src/2D_DFT/Constellation_Evolution.py # Complex plane symbol organization
python src/2D_DFT/Spectral_Attribution.py   # Frequency contribution to logits

# Circuit and geometry analyses
python src/circuit_Competition.py    # Algorithm vs memorization subspace
python src/intrinsic_dimension.py    # Intrinsic dimension estimation
python src/qk_circut.py              # QK circuit spectral analysis
```

### Visualization
```bash
# Generate all figures (data must exist first)
python src/plot/training_plot.py      # Training curves
python src/plot/2D_DFT/*.py           # All 2D DFT visualizations
python src/plot/circuit_Competition.py
python src/plot/intrinsic_dimension.py
# ... (see src/plot/ directory for all plotting scripts)
```

### Paper Compilation
```bash
# Compile complete paper with figures
bash compile_complete.sh

# Or manually:
pdflatex grokking_spectral_complete.tex
bibtex grokking_spectral_complete
pdflatex grokking_spectral_complete.tex
pdflatex grokking_spectral_complete.tex
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
├── src/
│   ├── grokking_reproduce.py          # Main training script
│   ├── 2D_DFT/                        # 2D DFT data collection
│   │   ├── psd.py                    # Power spectral density
│   │   ├── gini_ipr.py              # Gini + IPR (2D)
│   │   ├── f_alginment.py           # Frequency alignment
│   │   ├── Fourier_Projection.py    # Fourier sparsity
│   │   ├── Phase_Coherence.py       # Phase linearity
│   │   ├── Phase_Intervention.py    # Phase intervention
│   │   ├── Constellation_Evolution.py # Symbol organization
│   │   └── Spectral_Attribution.py  # Frequency attribution
│   ├── plot/                         # Visualization scripts
│   │   ├── 2D_DFT/                  # 2D DFT plots
│   │   └── [individual plot scripts]
│   ├── [1D analysis scripts]
│   └── [other analysis scripts]
├── data/x+y/
│   ├── checkpoints/                  # 1000 checkpoint files
│   ├── metric.csv                    # Training metrics
│   ├── gini_ip_2d.csv               # Sparsity data
│   ├── f_alginment_2d.csv           # Alignment data
│   ├── phase_intervention.csv       # Phase intervention
│   ├── circuit_Competition.csv      # Circuit competition
│   ├── spectral_attribution.csv    # Spectral attribution
│   └── [18 CSV files total]
├── experiments/figures/
│   ├── training_curves.png/pdf
│   ├── gini_ipr_2d.png/pdf
│   ├── f_alginment_2d.png/pdf
│   ├── phase_intervention.png/pdf
│   ├── circuit_Competition_detailed.png/pdf
│   ├── spectral_attribution_heatmap.png/pdf
│   ├── [24 PNG + 24 PDF files]
│   └── 2D_DFT/                       # 2D DFT specific figures
└── grokking_spectral_complete.tex   # Paper LaTeX source
```

### Core Analysis Scripts

Each analysis script follows the same pattern:

```python
# Data collection (src/XXX.py)
1. Load checkpoints from data/x+y/checkpoints/
2. Extract weights (W_E, W_U, attention matrices, etc.)
3. Compute analysis metrics (DFT, IPR, projections, etc.)
4. Save to data/x+y/XXX.csv

# Visualization (src/plot/XXX.py)
1. Load data/x+y/XXX.csv
2. Create matplotlib figures (multi-panel, log-scale, annotations)
3. Save to experiments/figures/XXX.png and XXX.pdf
```

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
| **Constellation** | Complex plane | No circular organization | constellation_data.csv |
| **Intrinsic Dimension** | Dimension | W_E decreases, W_U U-shaped | intrinsic_dimension.csv |

### Model Architecture (from grokking_reproduce.py)

The model must match this architecture for checkpoint compatibility:

```python
class GrokkingTransformer(nn.Module):
    - vocab_size = p + 1 = 98
    - embedding: nn.Embedding(98, 128)
    - pos_encoding: nn.Parameter(torch.randn(1, 3, 128))
    - 2 × TransformerBlock (d_model=128, n_heads=4, d_ff=512, dropout=0.1)
    - output: nn.Linear(128, 97)
```

**Critical hyperparameters:**
- `p = 97` (modulus)
- `weight_decay = 0.005` (enables Grokking)
- `total_steps = 100000`
- `save_interval = 100` (creates 1000 checkpoints)
- `train/test split = 50/50` (fixed seed 42)

### Paper Structure

The complete paper (`grokking_spectral_complete.tex`) includes:

**Main Text (6 pages):**
- Abstract + Title
- Introduction (motivation, related work, 5 contributions)
- Methods (6 analysis types with mathematical formulations)
- Results (6 key findings with 8 figures)
- Discussion (integrated perspective, theoretical implications)
- Conclusion
- References (17 papers)

**Appendix (2 pages):**
- 7 additional figures
- Complete results summary table

**Figures in main text:**
1. Training dynamics
2. Sparsification (IPR, Gini)
3. 2D Frequency alignment
4. Fourier domain sparsity
5. Phase coherence
6. Phase intervention
7. Circuit crossover
8. Spectral attribution heatmap

## Current Status

**Experiments completed:**
- ✅ Grokking reproduction (100K steps, 1000 checkpoints)
- ✅ All 1D DFT analyses (5 types)
- ✅ All 2D DFT analyses (8 types)
- ✅ Circuit competition analysis
- ✅ Intrinsic dimension analysis
- ✅ QK circuit spectral analysis
- ✅ All visualizations generated (48 figures)
- ✅ Complete paper written (8 pages with appendix)
- ✅ Deployed to GitHub

**Paper ready for submission** (needs author info, acknowledgments)

## Key Findings Summary

1. **Sparsification precedes generalization**: IPR peaks at 20K steps, 10K before test accuracy improves
2. **Circuit crossover drives transition**: Algorithm subspace overtakes memorization at 22K steps
3. **2D spectral structure is complex**: 2D alignment decreases (0.78→0.68), contradicting 1D results
4. **Phase structure is non-linear**: Ideal phase intervention destroys performance (100%→1%)
5. **No magic frequency**: Spectral attribution shows stable distribution across bands

## Important Notes

- **All checkpoint files are large**: Each .pt file is several MB, 1000 files total
- **Figure paths in LaTeX**: The paper uses `\graphicspath{{../experiments/figures/}{../experiments/figures/2D_DFT/}}` to locate figures
- **1D vs 2D DFT contradiction**: This is a critical finding that 1D analysis misses bidirectional spectral patterns
- **Phase intervention negative result**: The fact that "ideal" phases destroy performance is itself an important finding
- **Temporal resolution**: 100-step intervals enable precise correlation of metrics with Grokking onset

## For Paper Submission

1. Update author information in `grokking_spectral_complete.tex` (line 35-39)
2. Add acknowledgments with funding information
3. Review all figures are correctly displayed
4. Final compilation with `bash compile_complete.sh`
5. Create submission package with tex, bib, and figures folders
