# D8FN: Differentiable D8 Flow Routing with Physics Constraints for SAR Flood Mapping

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of the paper **"D8FN: Differentiable D8 Flow Routing with Physics Constraints for SAR Flood Mapping"** (submitted to IEEE JSTARS, 2026).

---

## Abstract

Synthetic Aperture Radar (SAR) flood mapping suffers from inherent ambiguity: smooth surfaces produce backscatter signatures nearly identical to floodwater, resulting in false positives in standard semantic segmentation models. We introduce **D8FN**, a physics-informed architecture that incorporates D8 topographic flow routing as a differentiable neural network layer and a Height Above Nearest Drainage (HAND) violation penalty in the loss function. Key results:

| Metric | 4-fold CV (mean ± std) | Cross-continent Test | 
|---|---|---|
| IoU (flood) | 0.575 ± 0.028 | 0.379 |
| F1 (flood) | 0.679 ± 0.023 | 0.466 |
| PA-IoU | 0.717 ± 0.020 | 0.640 |
| HVR | 1.00 ± 0.25% | 2.5% |

**Ablation contributions (Fold 0, per-tile Wilcoxon):**
- HAND violation penalty: **+0.072 IoU** (p < 0.001, Cohen's d = +0.257)
- D8 flow routing: **+0.027 IoU** (p < 0.01, Cohen's d = +0.091)

---

## Architecture

D8FN is a physics-informed encoder-decoder with three novel components:

```
SAR (VV, VH) + DEM + HAND + Slope + FlowDir + FlowAcc (9ch)
       |
  Input Projection (1x1 Conv → 3ch)
       |
  ConvNeXt-Small Encoder (ImageNet-22K pretrained)
       |
  ├─── Coarse D8 Routing (7×7, 50 rounds) ──┐
  │                                          ├── Fuse → Decoder → 
  └─── Fine D8 Routing (28×28, 25 rounds) ──┘         │
                                            ┌─────────┘
                                            ├── Height Field Head → H_w > DEM → Flood Logit
                                            └── Flood Logit Output
```

1. **D8FlowRouting Layer** — Differentiable message passing along D8-derived flow paths using learned sigmoid gates and scatter_reduce aggregation
2. **HAND Violation Penalty** — Physics-aware loss term: `mean(pred · 𝟙[HAND > 5m] · (1 − target))` penalizes only false positives in high-HAND areas
3. **Height Field Head** — Predicts water surface elevation H_w, thresholds against DEM: `flood_logit = (H_w − DEM) / τ`

---

## Repository Structure

```
d8fn-paper/
├── README.md
├── LICENSE
├── requirements.txt
│
├── d8fn/                          # Core package
│   ├── routing.py                 # ★ D8FlowRouting + D8FlowRoutingBlock
│   ├── models.py                  # D8FN architecture + baselines
│   ├── losses.py                  # Physics-constrained loss functions
│   ├── metrics.py                 # IoU, F1, PA-IoU, HVR, Betti, 3-class
│   ├── data.py                    # FloodDataset, augmentations, D8 computation
│   ├── train.py                   # Training loop, EMA, 5-fold CV, TTA
│   └── visualize.py               # Publication-quality figures
│
├── scripts/
│   ├── train_d8fn.py              # CLI training entry point
│   ├── eval_test_set.py           # Test set evaluation with TTA + threshold sweep
│   └── preprocess.py              # Data preprocessing: tortilla → .pt
│
├── notebooks/
│   ├── D8FN_Kaggle_v3.ipynb       # Full 4-fold CV training (Kaggle-ready)
│   └── D8FN_Ablation_Kaggle.ipynb # Ablation study (5 models, Fold 0)
│
├── paper/                         # LaTeX source
│   ├── paper_d8fn.tex
│   ├── references.bib
│   ├── architecture.md
│   └── figures/
│
├── figures/                       # High-resolution figures
│   ├── fig1_framework.svg
│   ├── fig2_routing_layer_detail.svg
│   ├── fig3_results_barchart.png
│   ├── fig4_radar.png
│   ├── fig_qualitative.png
│   ├── fig_confidence.png
│   ├── fig_multi_sample.png
│   └── fig_water_height.png
│
└── results/                       # Key result JSONs
    ├── D8FN_v3_4fold_results.json
    ├── ablation_results.json
    ├── test_eval_results.json
    └── physics_ablation/
```

---

## Quick Start

### Installation
```bash
git clone https://github.com/piyushdotcomm/d8fn.git
cd d8fn
pip install -r requirements.txt
```

### Training (Kaggle)
Upload the notebook `notebooks/D8FN_Kaggle_v3.ipynb` to Kaggle with:
- **Datasets:** processed SAR tiles, D8FN code folder, pretrained checkpoint (optional)
- **Accelerator:** GPU T4 × 2
- **Runtime:** ~11h for 4-fold CV, ~9h for ablation

### Evaluation
```bash
# Test set evaluation
python scripts/eval_test_set.py --checkpoint checkpoints/D8FN_fold0_best.pt --test-dir data/processed_v4_test --tta

# Compute ablation statistics
python scripts/train_d8fn.py --mode ablation --fold 0
```

### Reproducing Paper Results
1. Preprocess Kuro Siwo dataset: `python scripts/preprocess.py`
2. Train 4-fold CV: Run `D8FN_Kaggle_v3.ipynb` on Kaggle
3. Run ablation: Run `D8FN_Ablation_Kaggle.ipynb` on Kaggle
4. Evaluate test set: `python scripts/eval_test_set.py`
5. Compile paper: `cd paper && pdflatex paper_d8fn.tex`

---

## Dataset

We use the [Kuro Siwo dataset](https://github.com/Orion-AI-Lab/KuroSiwo) (NeurIPS 2024) via the [GEO-Bench-2](https://github.com/The-AI-Alliance/GEO-Bench-2) curated subset:
- **Training:** 5,000 SAR-DEM tile pairs (224×224, 10m resolution)
- **Test:** 2,000 tiles from 6 unseen flood events (Honduras, Malawi, Greece, Romania, France, Australia)
- **Labels:** 3-class expert photo-interpretation (no-water, permanent water, flood)

### Citation
When using Kuro Siwo, please cite:
> Bountos, N. I., Sdraka, M., Zavras, A., et al. "Kuro Siwo: 33 billion m² under the water. A global multi-temporal satellite dataset for rapid flood mapping." *NeurIPS Datasets & Benchmarks Track*, 2024.

---

## Key Results

### Ablation Study (Fold 0)
| Model | IoU | PA-IoU | HVR | ΔIoU | Wilcoxon p | Cohen's d |
|---|---|---|---|---|---|---|
| **D8FN (Full)** | **0.600** | **0.733** | **0.005** | — | — | — |
| − Physics Loss | 0.528 | 0.708 | 0.011 | +0.072 | <0.001** | +0.257 |
| − D8 Routing | 0.573 | 0.722 | 0.006 | +0.027 | 0.010** | +0.091 |
| UNet-R50 | 0.643 | 0.767 | 0.007 | −0.043 | 1.000 | −0.153 |
| DeepLabV3+-R50 | 0.617 | 0.756 | 0.008 | −0.017 | 0.952 | −0.060 |

**Top block:** same-backbone physics ablations. **Bottom block:** heterogeneous-backbone context. See paper for discussion.

### Routing Rounds Analysis
| Rounds (K) | IoU | PA-IoU | F1 |
|---|---|---|---|
| 10 | 0.581 | 0.718 | 0.690 |
| 25 | 0.575 | 0.707 | 0.686 |
| 50 (Full) | 0.600 | 0.733 | 0.688 |
| 100 | NaN (scatter overflow) | — | — |

---

## Metrics

- **IoU / F1:** Standard per-tile binary flood segmentation metrics
- **HVR (Height Violation Rate):** Fraction of predicted flood pixels with HAND > 5m
- **PA-IoU (Physics-Aware IoU):** mIoU × (1 − HVR) — combines accuracy with physics compliance

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details. The Kuro Siwo dataset is distributed under CC BY by its original authors.

---

## Citation

If you use D8FN in your research, please cite:

```bibtex
@article{kumar2026d8fn,
  title={D8FN: Differentiable D8 Flow Routing with Physics Constraints for SAR Flood Mapping},
  author={Kumar, Piyush and Prusty, Manas Ranjan},
  journal={submitted to IEEE JSTARS},
  year={2026}
}
```

---

## Authors

- **Piyush Kumar** — School of Computer Science, VIT Chennai, India
- - **Jay Gopal Tripathy** — School of Computer Science, VIT Chennai, India
- **Manas Ranjan Prusty** — School of Computer Science, VIT Chennai, India


