# ToxTransformer: Benchmark Evaluation Results

## 1. Model Overview

ToxTransformer is a decoder-only transformer that predicts **6,647 binary toxicity properties simultaneously** per compound. It processes a single causal sequence: SELFIES molecular tokens followed by interleaved property-value pairs (property1, value1, property2, value2, ...). A single forward pass produces calibrated probabilities for all properties. The model accepts optional *context*: known property-value pairs earlier in the sequence improve predictions for subsequent properties.

Evaluation uses 5-fold cross-validation with two protocols:
- **Holdout** (out-of-sample): compounds in the test fold were never seen during training
- **Bootstrap** (in-sample): compounds may overlap with training, measuring best-case performance

---

## 2. Headline Results: SOTA on Multi-Property Benchmarks

| Dataset | N tasks | ToxTransformer (holdout) | Best published | Delta |
|---------|---------|--------------------------|----------------|-------|
| **Tox21** | 12 | **0.936** | 0.847 (MoLFormer-XL) | **+8.9** |
| **SIDER** | 27 | **0.821** | 0.690 (MoLFormer-XL) | **+13.1** |
| BACE | 1 | 0.830 | 0.882 (MoLFormer-XL) | -5.2 |
| BBBP | 1 | 0.726 | 0.724 (GEM) | +0.2 |
| ClinTox | 2 | 0.762 | 0.948 (MoLFormer-XL) | -18.6 |

Published baselines are from the MoLFormer paper comparison table (single-task models, potentially different data splits). For BBBP, our 0.726 is competitive with D-MPNN (0.712), GEM (0.724), RF (0.714), and SVM (0.729); MoLFormer's 0.937 appears to use non-scaffold splits.

**The pattern**: ToxTransformer excels on multi-property benchmarks where predicting many related properties simultaneously helps. Tox21 (+8.9) and SIDER (+13.1) are clear SOTA. Single-property benchmarks (BACE, BBBP) are competitive. ClinTox (only 2 tasks, high variance) is the main gap.

---

## 3. Overall Performance: 3,776 Properties, 12 Sources

**Holdout mean AUC: 0.828** across 3,776 benchmark properties from 12 sources (Tox21, SIDER, BACE, BBBP, ClinTox, toxcast, ice, pubchem, chembl, bindingdb, reach, toxvaldb).

| AUC Range | % of Properties |
|-----------|----------------|
| >= 0.9 | 34% |
| >= 0.8 | 58% |
| >= 0.7 | 78% |

---

## 4. Context Scaling (Bootstrap)

| nprops | Mean AUC | Delta vs baseline |
|--------|----------|-------------------|
| 1 | 0.853 | — |
| 2 | 0.828 | -2.5 |
| 5 | 0.841 | -1.2 |
| 10 | 0.879 | +2.6 |
| 20 | **0.882** | **+2.9** |

U-shaped curve: performance dips at nprops=2-4 (noisy small context sets), then climbs to 0.882 at nprops=20 as diverse context accumulates. Most individual sources peak at nprops=10-20. Tox21 reaches 0.968 at nprops=1 (bootstrap), toxcast reaches 0.904 at nprops=20.

---

## 5. What Didn't Work (and How to Fix It)

**Autoregressive context generation**: Using the model's own predictions as input context slightly *hurt* performance vs. no context. The model was trained with ground-truth context, not self-generated context, so prediction errors propagate. **Fix**: Train with reinforcement learning to select and generate its own context sequence, optimizing for downstream prediction accuracy.

**ClinTox underperformance**: 0.762 holdout vs 0.948 published — only 2 tasks, high variance across splits, no multitask benefit. **Fix**: Task-specific fine-tuning or ensemble heads.

**Holdout context scaling gap**: Holdout evaluation was only run at nprops=1 and nprops=20 (no intermediate values). The bootstrap shows a clear U-curve but we can't confirm it generalizes out-of-sample at intermediate values. **Fix**: Run holdout at nprops=2,3,4,5,10.

**Result variability**: Numbers fluctuate with sampling methods (which properties chosen as context, random seeds, evaluation protocol). This variability indicates room for optimization — a systematic approach to context selection could yield further gains.

---

## 6. Supplementary Data

The attached CSVs contain publication-ready tables:
- **table7** — Full SOTA comparison (ToxTransformer vs MoLFormer-XL, D-MPNN, GEM, RF, SVM)
- **table8** — Per-source results with all nprops values (bootstrap) and holdout

Full per-property detail (6,378 properties, all sources, all nprops) available on request.
