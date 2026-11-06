# OCT Anomaly Detection Protocol

One-class anomaly detection on retinal OCT B-scans (NEH dataset, Noor Eye Hospital: Normal / Drusen / CNV).
Every model is trained **only on normal B-scans** and flags Drusen and CNV scans as anomalies. The full
protocol is in [PRD.md](PRD.md); every deviation from it is logged in
[outputs/logs/changes.md](outputs/logs/changes.md).

## Results (test set, never used before script 10)

Patient-level split (84 / 18 / 18 NORMAL patients, 0 / 40 / 84 DRUSEN, 0 / 40 / 86 CNV for train / val / test).
95% CIs from 1,000 bootstrap resamples of test **patients**. Case = patient, scored by the max over its B-scans.

| Method | B-scan ROC-AUC | B-scan PR-AUC (baseline 0.656) | Case ROC-AUC | Case PR-AUC (baseline 0.904) |
|---|---|---|---|---|
| M1 Convolutional autoencoder | 0.570 [0.516, 0.621] | 0.700 | 0.503 [0.358, 0.634] | 0.911 |
| **M2 Deep SVDD, frozen ResNet18** | **0.672 [0.638, 0.709]** | **0.815** | **0.814 [0.751, 0.872]** | **0.979** |
| M2 Deep SVDD, fine-tuned | 0.522 [0.492, 0.553] | 0.673 | 0.443 [0.339, 0.549] | 0.914 |
| M3 Convolutional VAE | **0.674 [0.628, 0.721]** | 0.807 | 0.722 [0.637, 0.801] | 0.965 |
| M4 MKD (100 epochs) | 0.629 [0.584, 0.679] | 0.773 | 0.663 [0.566, 0.757] | 0.956 |

At the primary threshold (95th percentile of validation-normal scores), frozen Deep SVDD detects 73% of CNV
patients and 19% of Drusen patients at 100% specificity on the 18 NORMAL test patients.
Full table: [outputs/metrics/comparison_table.md](outputs/metrics/comparison_table.md); ablations:
[outputs/metrics/ablations.md](outputs/metrics/ablations.md).

**Main findings**
- **CNV is detectable, Drusen is not** (score distributions: `outputs/figures/score_distributions.png`).
  Drusen are small sub-retinal deposits that barely change global appearance; every method scores them
  like normal B-scans.
- **Frozen ImageNet features are the most reliable.** Frozen Deep SVDD is best per patient and ties the VAE per
  B-scan. Fine-tuning Deep SVDD with the one-class loss collapses the embedding (val mean distance ~3e-4) and is
  unstable: retrained on the GPU it fell to chance level (case ROC-AUC 0.443, previously 0.549).
- **The autoencoder reconstructs lesions too well** (`outputs/figures/ae_heatmaps/`), so its residual is weak.
  The VAE's constrained latent space avoids this: it is the best B-scan detector (0.674) and the best for CNV
  slices (normal vs CNV validation ROC-AUC 0.908).
- **Leakage inflates results for every model:** an image-level random split raises the case ROC-AUC of the
  autoencoder from 0.503 to 0.648, frozen SVDD from 0.814 to 0.877 and fine-tuned SVDD from 0.443 to 0.816
  (`ablations.md`, retraining ablations).
- **Adding normal-labelled slices from diseased eyes** to training (expanded pool) changes nothing measurable.
- **The 7 excluded layout-outlier patients** (a different export format, abnormal classes only) would be
  flagged 82-100% of the time by SVDD, the VAE and MKD - a shortcut the exclusion keeps out of the main table.

## Improvements after the main study (still one-class)

Added later, outside the original protocol ([changes.md](outputs/logs/changes.md) items 27-28; full tables in
[outputs/improvements/results.md](outputs/improvements/results.md) and
[outputs/external/kermany_results.md](outputs/external/kermany_results.md)). Every model is still fitted on
clean-normal train B-scans only, and every choice was made on validation before the test set was read.

- **M5 PatchCore-style detector** (`16_patchcore.py`): frozen ResNet18 patch features, a memory bank of healthy
  train patches, score = the most unusual patch. No training.
- **Map-max scoring** (`15_external_kermany.py`): score = peak of the anomaly map instead of its average (Aresta et al.).
- **Ensemble + neighbouring-slice smoothing** (`17_improvements_report.py`): z-scored models combined, then a 5-slice
  moving mean within each eye.

| Method (NEH test) | B-scan ROC-AUC | Drusen AUC | CNV AUC | Case ROC-AUC [95% CI] |
|---|---|---|---|---|
| Previous best (M2 frozen SVDD / M3 VAE) | 0.672 / 0.674 | 0.551 / 0.541 | 0.867 / 0.890 | 0.814 [0.751, 0.875] / 0.722 |
| M5 PatchCore | 0.748 | 0.635 | 0.930 | 0.813 [0.724, 0.893] |
| M5 PatchCore + slice smoothing | 0.777 | 0.666 | 0.955 | 0.803 [0.718, 0.881] |
| **Ensemble + slice smoothing (val-chosen final)** | **0.771** | **0.661** | **0.947** | **0.834 [0.754, 0.904]** |

**Comparison with published work.**
- *Sotoudeh-Paima et al. 2022*, the NEH dataset paper: a supervised FPN-VGG16 trained with disease labels reaches
  92.0% accuracy (sensitivity 91.8%, specificity 95.8%). On the same worst-case B-scan subset of our test patients,
  our best one-class model reaches ROC-AUC 0.82, with sensitivity 0.44 at specificity 0.97. A model trained on the
  diseases remains far better.
- *Aresta et al. 2025* (label-free) on the Kermany test set: 0.87 overall, 0.70 on drusen. Our val-chosen PatchCore
  reaches 0.855 overall and 0.727 on drusen. The other pre-registered variant (layer 2+3), not chosen on validation,
  reaches 0.918.

## Data findings (EDA)

See [outputs/eda/eda_report.md](outputs/eda/eda_report.md) and `outputs/eda/figures/`.
- 16,803 B-scans, 441 patient folders (120 / 160 / 161); case numbers restart in every class folder.
- `data_information.csv` agrees with the filenames for every file; its 16,822 rows = 16,803 files + 12
  duplicate rows + 7 files lost to a case-insensitive filename clash (`DRUSEN/59`).
- 4 pairs of patient folders contain byte-identical volumes (kept, per decision; listed as a limitation).
- 7 patients use a different image layout (excluded from the main experiment, scored separately).
- All images are 768 x 496 grayscale stored as RGB, with a burned-in scale bar (masked in preprocessing).

## Pipeline

| Script | Purpose |
|---|---|
| `01_dataset_check.py`, `01b_eda.py` | read-only inspection and exploratory analysis |
| `02_build_manifest.py`, `03_split_data.py` | manifest (one row per B-scan) and patient-level splits |
| `04_preprocess.py` | retina ROI crop (352 rows, full width), scale-bar mask, 224 x 224 cache |
| `05` / `06` | M1 autoencoder: train / score |
| `07` / `08` | M2 Deep SVDD (frozen and fine-tuned): train / score |
| `09_train_vae.py` | M3 VAE: train and score |
| `12_train_mkd.py` | M4 multiresolution knowledge distillation: train and score |
| `10_compare_models.py` | the only script that computes test metrics; bootstrap CIs, ablations |
| `11_visualize_results.py` | ROC / PR curves, score distributions, confusion matrices, heatmaps |
| `13_ablation_retrain.py` | expanded-pool and naive-split ablations |
| `14_audit.py` | automated checks of data, splits, saved models and metrics |
| `15_external_kermany.py` | external test on Kermany (no retraining) + map-max scores |
| `16_patchcore.py` | M5 patch-level nearest-neighbour detector (frozen ResNet18, no training) |
| `17_improvements_report.py` | ensemble, slice smoothing, comparison with Sotoudeh-Paima 2022 and Aresta 2025 |

`common.py` holds shared helpers (data access, splits, dataset, case aggregation); `nets.py` holds the networks.

## Demo in VS Code

1. Open the project folder in VS Code (File > Open Folder). `.vscode/settings.json` selects the `.venv-gpu` Python.
2. **Walkthrough:** open `notebooks/project_walkthrough.ipynb`. It is saved with all outputs, so every stage is
   visible immediately; **Run All** (kernel: `.venv-gpu`) re-runs everything live in about 3 minutes - dataset
   checks, EDA, the leak-free split, preprocessing on a raw scan, a 2-epoch training demo, live scoring of three
   test scans by all four models with heatmaps, validation and test results, ablations and the audit.
3. **Individual stages:** Terminal > Run Task... lists every pipeline script (dataset check, EDA, manifest, split,
   preprocessing, training demo, scoring, comparison, figures, audit, full re-run).
4. **Colab:** `notebooks/04_preprocessing_colab.ipynb` runs the preprocessing step by step on Google Colab.

## Setup and running

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```
For a GPU, install PyTorch from the CUDA index instead (e.g. `--index-url https://download.pytorch.org/whl/cu126`).
Set `data_root` in `config.yaml` to the extracted dataset folder **or** the Mendeley download zip (it is read
in place). Then run the scripts in numeric order from the project root. `03` refuses to overwrite
`outputs/splits.csv` without `--force`.

## Limitations

- Single site and device; no external validation. Each B-scan is scored independently, without 3D context.
- Duplicate patient folders were kept (4 pairs); one pair spans val and test.
- All models were trained on a 4 GB GPU in full float32. MKD was interrupted at epoch 49 (disk full) and resumed
  from that checkpoint with a fresh optimizer state (`changes.md` item 26).
- Fine-tuned Deep SVDD is unstable across retraining (see findings); only one run per model was made, so no
  seed-to-seed spread is reported.
- The score flags abnormality, not a specific disease: a screening aid, not a diagnosis.

## References

1. Sotoudeh-Paima S. et al. Labeled Retinal OCT Dataset for Classification of Normal, Drusen, and CNV Cases. Mendeley Data v2.
2. Ruff L. et al. Deep One-Class Classification. ICML 2018.
3. Salehi M. et al. Multiresolution Knowledge Distillation for Anomaly Detection. CVPR 2021.
4. Aresta G. et al. Anomaly Detection in Retinal OCT Images With Deep Learning-Based Knowledge Distillation. TVST 14(3):26, 2025.
5. Zhou X. et al. Spatial-contextual variational autoencoder with attention correction for anomaly detection in retinal OCT images. Comput. Biol. Med. 152:106328, 2023.
6. Schlegl T. et al. Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery (AnoGAN). IPMI 2017.
