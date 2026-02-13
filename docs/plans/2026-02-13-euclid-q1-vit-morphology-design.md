# Euclid Q1 ViT Galaxy Morphology — Project Design

**Date**: 2026-02-13
**Author**: Alexy Louis
**Status**: Approved

## Summary

Apply Vision Transformers to Euclid Q1 galaxy morphology data, regressing on
Galaxy Zoo vote fractions. Benchmark 6 architectures (Zoobot, ConvNeXt, ViT-Base,
Swin-V2, DINOv2, DINOv3) against the published Zoobot baseline. Include attention
map vs GradCAM interpretability comparison. Target A&A or MNRAS publication.

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Task formulation | Regression on vote fractions | Matches Zoobot; preserves morphology uncertainty |
| Loss function | Dirichlet-Multinomial | Direct comparison with Walmsley et al. 2025 |
| Model count | 6 (2 CNN + 4 Transformer) | Focused enough for clean paper narrative |
| Interpretability | Attention rollout + GradCAM | Side-by-side ViT vs CNN comparison |
| Evaluation | Zoobot-aligned metrics | Bulletproof comparison for reviewers |
| TTA | 7 augmented views at inference | Free performance boost, no training cost |
| Git strategy | dev branch + tagged milestones | Clean main, flexible dev iteration |
| Data scope | 50K galaxies, expandable to 150K | Publication-sufficient, storage-friendly |
| Environment | VSCode + Colab extension hybrid | Local dev, GPU training via Colab |

## Project Phases

1. **v0.1-eda** — Catalog exploration, sample visualization, quality filtering
2. **v0.2-data** — Dataset classes, transforms, stratified splits
3. **v0.3-baseline** — Reproduce Zoobot metrics on Euclid Q1
4. **v0.4-vit** — Train all 6 models with linear probe + full fine-tune
5. **v0.5-benchmark** — Comparison tables, statistical significance, TTA
6. **v0.6-interp** — Attention maps, GradCAM, failure analysis
7. **v0.7-figures** — Publication-ready plots (PDF/SVG)
8. **v0.8-paper** — Results interpretation, paper draft
9. **v1.0-submission** — Code cleanup, reproducibility check, submit

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| DINOv3 not officially released | Use Kaggle models; paper still strong with DINOv2 |
| Zoobot hard to reproduce exactly | Use their public weights + adapt to Euclid Q1 |
| 50K galaxies insufficient | Expandable to 150K via additional ZIP batches |
| ViTs don't beat CNNs | Paper reframes as "comprehensive benchmark" — still publishable |
