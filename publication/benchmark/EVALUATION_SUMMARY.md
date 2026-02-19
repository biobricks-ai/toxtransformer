# ToxTransformer: Benchmark Evaluation Results

## 1. Model Overview

ToxTransformer is a decoder-only transformer that predicts **6,647 binary toxicity properties simultaneously** per compound. It processes a single causal sequence: SELFIES molecular tokens followed by interleaved property-value pairs (property1, value1, property2, value2, ...). A single forward pass produces calibrated probabilities for all properties. The model accepts optional *context*: known property-value pairs earlier in the sequence improve predictions for subsequent properties.

Evaluation uses 5-fold cross-validation with two protocols:
- **TT0 / Holdout** (out-of-sample, no context): compounds in the test fold were never seen during training
- **TT20 / Bootstrap** (in-sample, 20 context properties): compounds may overlap with training, measuring best-case performance with context

---

## 2. Headline Results: TT0 vs TT20 vs Best Published

| Dataset | N tasks | TT0 (holdout) | TT20 (bootstrap) | Best Published | TT0 delta | TT20 delta |
|---------|---------|----------------|-------------------|----------------|-----------|------------|
| **Tox21** | 8 | **0.957** | **0.993** | 0.847 (MoLFormer-XL) | **+11.0** | **+14.6** |
| **SIDER** | 26 | **0.762** | **0.886** | 0.690 (MoLFormer-XL) | **+7.2** | **+19.6** |
| BACE | 1 | 0.827 | — | 0.882 (MoLFormer-XL) | -5.5 | — |
| **BBBP** | 1 | **0.866** | **0.858** | 0.724 (GEM) | **+14.2** | **+13.4** |
| ClinTox | 2 | 0.786 | 0.804 | 0.948 (MoLFormer-XL) | -16.2 | -14.4 |

Published baselines are from the MoLFormer paper comparison table (single-task models, potentially different data splits). TT0 = holdout, no context. TT20 = bootstrap, 20 context properties. BACE has no TT20 (single property, no context available).

**The pattern**: ToxTransformer excels on multi-property benchmarks. Context amplifies this — SIDER jumps from +7.2 to +19.6 with 20 context properties. Tox21 and BBBP are clear wins in both settings. ClinTox (only 2 tasks, high variance) is the main gap.

---

## 3. Overall Performance: 2,227 Properties, 11 Sources

**Holdout median AUC: 0.833** across 2,227 benchmark properties from 11 sources (tox21, SIDER, BACE, BBBP, ClinTox, toxcast, ice, pubchem, chembl, reach, toxvaldb).

With 20 context properties (bootstrap): **median AUC: 0.936**.

Across all 6,378 holdout-evaluated properties (including non-benchmark sources):

| AUC Range | % of Properties |
|-----------|----------------|
| >= 0.9 | 31% |
| >= 0.8 | 64% |
| >= 0.7 | 85% |

---

## 4. Context Scaling (Bootstrap, 2,227 properties)

| nprops | Median AUC | Delta vs baseline |
|--------|------------|-------------------|
| 1 | 0.862 | — |
| 2 | 0.868 | +0.6 |
| 3 | 0.870 | +0.8 |
| 4 | 0.875 | +1.3 |
| 5 | 0.879 | +1.7 |
| 10 | 0.910 | +4.8 |
| 20 | **0.936** | **+7.4** |

Monotonic improvement with context. Most sources show clear gains at nprops=10-20: pubchem 0.803 to 0.976, sider 0.783 to 0.886, chembl 0.863 to 0.966.

---

## 5. Per-Source Results (all 11 sources)

| Dataset | N tasks | TT0 (holdout) | TT20 (bootstrap) |
|---------|---------|----------------|-------------------|
| tox21 | 8 | 0.957 | 0.993 |
| ice | 376 | 0.926 | 0.938 |
| toxcast | 323 | 0.910 | 0.909 |
| BBBP | 1 | 0.866 | 0.858 |
| toxvaldb | 37 | 0.833 | 0.884 |
| chembl | 381 | 0.833 | 0.966 |
| BACE | 1 | 0.827 | — |
| ClinTox | 2 | 0.786 | 0.804 |
| pubchem | 1070 | 0.775 | 0.976 |
| reach | 2 | 0.766 | 0.829 |
| sider | 26 | 0.762 | 0.886 |
| **Overall** | **2227** | **0.833** | **0.936** |

---

## 6. What Didn't Work (and How to Fix It)

**Autoregressive context generation**: Using the model's own predictions as input context slightly *hurt* performance vs. no context. The model was trained with ground-truth context, not self-generated context, so prediction errors propagate. **Fix**: Train with reinforcement learning to select and generate its own context sequence, optimizing for downstream prediction accuracy.

**ClinTox underperformance**: 0.786 holdout vs 0.948 published — only 2 tasks, high variance across splits, no multitask benefit. **Fix**: Task-specific fine-tuning or ensemble heads.

**Holdout context scaling gap**: Holdout evaluation was only run at nprops=1 and the 11-20 context bucket (no intermediate values). **Fix**: Run holdout at nprops=2,3,4,5,10.

---

## 7. Supplementary Data

The attached CSVs contain publication-ready tables:
- **table7** — TT0 vs TT20 vs Best Published comparison
- **table8** — Per-source results with all nprops values (bootstrap) and holdout

Full per-property detail (6,378 properties, all sources, all nprops) available on request.
