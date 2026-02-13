# CLAUDE.md — Euclid Q1 ViT Galaxy Morphology

## Project Overview

ViT-based galaxy morphology regression on Euclid Q1 data (Zenodo #15020547).
Regress Galaxy Zoo vote fractions using Vision Transformers, benchmarked against
Zoobot (Walmsley et al. 2025). Target publication: A&A or MNRAS.

## Environment

- **Local**: VSCode with Colab extension, Python 3.10+
- **GPU training**: Google Colab Pro (T4/V100/A100) via VSCode Colab extension
- **OS**: Windows 10

## Project Structure

```
euclid_q1/
├── CLAUDE.md
├── .gitignore
├── requirements.txt
├── configs/                  # Training configs (YAML)
│   └── base.yaml
├── data/
│   ├── raw/                  # Downloaded catalog + image batches
│   ├── processed/            # Preprocessed cutouts, splits
│   └── sample/               # Small subset for local dev (~100 images)
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis
│   ├── 02_data_preparation.ipynb       # Download, preprocess, split
│   ├── 03_baseline_zoobot.ipynb        # Zoobot reproduction
│   ├── 04_vit_experiments.ipynb        # ViT/Swin/DINOv2 training
│   ├── 05_benchmarking.ipynb           # Model comparison tables
│   └── 06_attention_analysis.ipynb     # Attention maps + GradCAM
├── src/
│   ├── data/                 # Dataset classes, transforms, downloading
│   ├── models/               # Model wrappers (Zoobot, ViT, Swin, DINOv2, ConvNeXt)
│   ├── training/             # Training loops, schedulers, loss functions
│   ├── evaluation/           # Metrics, calibration, comparison utilities
│   └── visualization/        # Attention maps, GradCAM, plots
├── scripts/                  # CLI entry points for Colab/batch runs
├── results/                  # Saved metrics, predictions, figures
│   ├── checkpoints/          # Model weights (gitignored)
│   ├── figures/              # Publication-ready plots
│   └── tables/               # CSV/LaTeX comparison tables
└── docs/
    └── plans/                # Design documents
```

## Models

| ID       | Architecture    | Type            | Pre-training          | Role                     |
|----------|-----------------|-----------------|-----------------------|--------------------------|
| zoobot   | EfficientNet-B0 | CNN             | GZ DECaLS             | Published baseline       |
| convnext | ConvNeXt-Base   | CNN             | ImageNet-21k          | Modern CNN reference     |
| vit-base | ViT-Base/16     | Transformer     | ImageNet-21k          | Vanilla ViT baseline     |
| swin-v2  | Swin-V2-Base    | Hybrid          | ImageNet-21k          | Hierarchical transformer |
| dinov2   | ViT-B/14        | SSL Transformer | LVD-142M (DINO)       | Self-supervised SOTA     |
| dinov3   | TBD             | SSL Transformer | Kaggle (unofficial)   | Latest SSL (if usable)   |

## Training Configuration

- **Task**: Regression on Galaxy Zoo vote fractions (per-question)
- **Loss**: Dirichlet-Multinomial (matching Zoobot) + MSE fallback
- **Optimizer**: AdamW, lr=5e-5 (fine-tune), lr=1e-3 (linear probe)
- **Scheduler**: Cosine annealing with linear warmup (5% of steps)
- **Batch size**: 32 (Colab T4) / 64 (Colab A100)
- **Epochs**: 30 with early stopping (patience=5, monitor val_loss)
- **Input size**: 224×224 RGB (resized from Euclid cutouts)
- **Augmentation**: Random horizontal/vertical flip, rotation (0-360°),
  color jitter (brightness=0.1, contrast=0.1), random crop+resize
- **Fine-tuning strategy**:
  1. Linear probe (frozen backbone, 5 epochs) → establish baseline
  2. Full fine-tune (unfreeze all, 25 epochs) → final performance
- **Reproducibility**: Seed all at 42, log all hyperparams to configs/

## Evaluation

### Metrics (Zoobot-aligned)

- **Primary**: Dirichlet-Multinomial loss per morphology question
- **Regression**: MSE, MAE, R² on vote fractions (per-question breakdown)
- **Correlation**: Pearson and Spearman per morphology question
- **Classification**: Accuracy, weighted F1 when discretizing predictions
  (argmax of predicted fractions → class label)
- **Calibration**: Predicted vs. true vote fraction reliability diagrams

### Test-Time Augmentation (TTA)

Apply at inference (no additional training cost):
- Original + horizontal flip + vertical flip + 90°/180°/270° rotations
- Average predicted vote fractions across augmented views
- Report both with-TTA and without-TTA results

### Comparison Framework

All models evaluated on the same held-out test set (80/10/10 split, stratified
by dominant morphology class). Results reported as:

1. **Summary table**: All models × all metrics (LaTeX-ready)
2. **Per-question breakdown**: Which morphology questions each model handles
   best/worst (smooth vs featured, edge-on, spiral arms, bar, bulge, etc.)
3. **Statistical significance**: Paired bootstrap confidence intervals on
   metric differences between models
4. **Inference speed**: Throughput (images/sec) and parameter count

### Interpretability

- **ViT models**: Attention rollout maps overlaid on galaxy cutouts
- **CNN models**: GradCAM heatmaps on final convolutional layer
- **Side-by-side figures**: Same galaxies, ViT attention vs CNN GradCAM
- **Failure case analysis**: Where do models disagree most? Show examples

## Data

### Source

- **Zenodo record**: 15020547 (Galaxy Zoo Euclid Q1)
- **Catalog**: morphology_catalogue.parquet (~50 MB)
- **Images**: ZIP batches on Zenodo, selectively extracted
- **Target sample**: 50K galaxies (stratified by morphology), expandable to 150K

### Data Split

- Train: 80% | Validation: 10% | Test: 10%
- Stratified by dominant morphology class
- Split saved as data/processed/split_indices.json (reproducible)
- NEVER leak test set into training or hyperparameter tuning

### Vote Fraction Targets

Galaxy Zoo decision tree questions (predict all simultaneously):
- smooth-or-featured (smooth / featured / artifact)
- disk-edge-on (yes / no)
- has-spiral-arms (yes / no)
- bar (yes / no)
- bulge-size (dominant / large / moderate / small / none)
- merging (merging / tidal / both / neither)

## Common Commands

### Local development

```bash
pip install -r requirements.txt                    # Setup environment
python -m pytest tests/                            # Run tests
python scripts/download_catalog.py                 # Fetch catalog only
python scripts/prepare_splits.py                   # Generate train/val/test
```

### Training (Colab via VSCode extension)

```bash
python scripts/train.py --config configs/dinov2.yaml          # Single model
python scripts/train.py --config configs/dinov2.yaml --tta    # With TTA eval
```

### Evaluation

```bash
python scripts/benchmark.py --results-dir results/            # Full comparison
python scripts/visualize_attention.py --model dinov2 --n 50   # Attention maps
```

### Figures

```bash
python scripts/generate_figures.py --format pdf    # Publication figures
```

## Git Strategy

- **main**: Clean milestones only, merged from dev at phase boundaries
- **dev**: Daily work, experiments, iteration
- **Tagging convention**: v0.X-<phase-name>

### Project Phases & Milestones

| Tag             | Phase                     | Deliverables                            |
|-----------------|---------------------------|-----------------------------------------|
| v0.1-eda        | Exploratory Data Analysis | 01_eda.ipynb, catalog stats, sample viz |
| v0.2-data       | Data Pipeline             | Dataset classes, transforms, splits     |
| v0.3-baseline   | Zoobot Baseline           | Reproduced Zoobot metrics on Euclid Q1  |
| v0.4-vit        | ViT Experiments           | All 6 models trained and logged         |
| v0.5-benchmark  | Benchmarking              | Comparison tables, stat tests, TTA      |
| v0.6-interp     | Interpretability          | Attention maps, GradCAM, failure cases  |
| v0.7-figures    | Publication Figures       | All plots publication-ready (PDF/SVG)   |
| v0.8-paper      | Paper Draft               | Results interpretation, draft sections  |
| v1.0-submission | Submission                | Final code cleanup, reproducibility     |

### Commit Convention

```
<type>(<scope>): <description>

Types: feat, fix, data, exp, viz, docs, refactor
Scopes: eda, pipeline, zoobot, vit, swin, dinov2, dinov3, convnext, bench, interp

Examples:
  feat(pipeline): add EuclidDataset with stratified splits
  exp(dinov2): linear probe achieves R²=0.87 on smooth fraction
  viz(interp): add side-by-side attention vs GradCAM figure
  data(eda): catalog exploration and quality filtering
```

## Key References

- Walmsley et al. 2025 — Galaxy Zoo Euclid Q1 morphology (Zoobot baseline)
- Oquab et al. 2024 — DINOv2: Learning Robust Visual Features
- Liu et al. 2022 — Swin Transformer V2
- Liu et al. 2022 — ConvNeXt: A ConvNet for the 2020s
- Dosovitskiy et al. 2021 — An Image is Worth 16x16 Words (ViT)
- Cao et al. 2024 — CvT galaxy morphology (A&A), 98.8% on 5-class
- Lastufka et al. 2024 — DINOv2 on astronomical images (SigLIP comparison)

## Guardrails

### Do

- Always seed everything (42) for reproducibility
- Log all hyperparameters and metrics to configs/ and results/
- Save predictions (not just metrics) for post-hoc analysis
- Use the same preprocessing pipeline across ALL models
- Report confidence intervals, not just point estimates
- Keep data/sample/ with ~100 images for fast local iteration

### Don't

- Never train or tune on the test set
- Never cherry-pick attention maps — show random samples + failures
- Never compare models trained with different augmentation pipelines
- Never commit data files, checkpoints, or API keys
- Never hardcode absolute paths — use relative paths or config variables

### Publication Checklist (before v1.0)

- [ ] All experiments reproducible from configs/ alone
- [ ] Comparison table includes parameter count + inference speed
- [ ] Attention/GradCAM figures use identical galaxies across models
- [ ] Statistical significance tested (bootstrap CI)
- [ ] Code cleaned, documented, and runnable by a reviewer
- [ ] Figures exported as vector (PDF/SVG) at 300+ DPI
