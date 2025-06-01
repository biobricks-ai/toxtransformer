
Multi-expert encoder-decoder model trained on ChemHarmony to predict chemical toxicity

A presentation of this work is available on google drive [ChemHarmony and Biosim](https://docs.google.com/presentation/d/1Krz6eh7ooOLe84m01M2C_3Q9-qsFZ4L9L3IixzDIQWc/edit)


# Evaluation
We want to evaluate this model using the same benchmarks from the [molformer](https://doi.org/10.1038/s42256-022-00580-7) paper and using some new benchmarks for the NIEHS acute inhalation project and Tox24:

1. [ ] SIDER
2. [ ] Tox21
3. [ ] ClinTox
4. [ ] HIV Activity
5. [ ] BACE
6. [ ] CoMPAIT - Collaborative Modeling Project for Acute Inhalational Toxicity

and for regression tasks
1. [ ] QM9 (quantum mechanical properties)
2. [ ] ESOL (solubility from moleculenet)
3. [ ] FreeeSolv (free solvation energy from moleculenet?)
4. [ ] lipophilicity (lipophilicity from moleculenet)
5. [ ] Tox24

| Model           | BBBP  | Tox21 | ClinTox | HIV   | BACE  | SIDER |
|-----------------|-------|-------|---------|-------|-------|-------|
| RF              | 71.4  | 76.9  | 71.3    | 78.1  | 86.7  | 68.4  |
| SVM             | 72.9  | 81.8  | 66.9    | 79.2  | 86.2  | 68.2  |
| MGCN            | 85.0  | 70.7  | 63.4    | 73.8  | 73.4  | 55.2  |
| D-MPNN          | 71.2  | 68.9  | 90.5    | 75.0  | 85.3  | 63.2  |
| DimeNet         | -     | 78.0  | 76.0    | -     | -     | 61.5  |
| Hu et al.       | 70.8  | 78.7  | 78.9    | 80.2  | 85.9  | 65.2  |
| N-gram          | 91.2  | 76.9  | 85.5    | 83.0  | 87.6  | 63.2  |
| MolCLR          | 73.6  | 79.8  | 93.2    | 80.6  | 89.0  | 68.0  |
| GraphMVP-C      | 72.4  | 74.4  | 77.5    | 77.0  | 81.2  | 63.9  |
| GeomGCL         | -     | 85.0  | 91.9    | -     | -     | 64.8  |
| GEM             | 72.4  | 78.1  | 90.1    | 80.6  | 85.6  | 67.2  |
| ChemBERTa       | 64.3  | -     | 90.6    | 62.2  | -     | -     |
| **MoLFormer-XL**| **93.7** | **84.7** | **94.8** | **82.2** | **88.21** | **69.0** |

* Bold indicates the top-performing model. 
* '—' signifies that the values were not reported for the corresponding task.

| Model           | QM9 (MAE) | QM8 (MAE) | ESOL (RMSE) | FreeSolv (RMSE) | Lipophilicity (RMSE) |
|-----------------|------------|-----------|-------------|-----------------|----------------------|
| GC              | 4.3536     | 0.0148    | 0.970       | 1.40            | 0.655                |
| A-FP            | 2.6355     | 0.0282    | 0.5030      | 0.736           | 0.578                |
| MPNN            | 3.1898     | 0.0143    | 0.58        | 1.150           | 0.7190               |
| **MoLFormer-XL**| **1.5894** | **0.0102**| **0.2787**  | **0.2308**      | **0.5289**           |

* Bold indicates the top-performing model.


## License
MIT
