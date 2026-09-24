# PRD — One-Class Anomaly Detection on Retinal OCT (NEH dataset)

> **How to use this file:** Put it in the project root as `PRD.md` and tell Claude Code:
> *"Read PRD.md. Build the project one script at a time in the order in §9. After each script, run it, show me the output, and wait for my go-ahead before starting the next one."*

---

## 1. Summary

Build a leakage-safe PyTorch pipeline that trains anomaly detectors **only on normal retinal OCT B-scans** and flags Drusen and CNV scans as anomalies. The pipeline compares three methods on identical patient-level splits:

| # | Method | Role |
|---|---|---|
| M1 | Convolutional autoencoder (from scratch) | Baseline |
| M2 | Deep SVDD on a pretrained ResNet18 encoder | **Primary method** |
| M3 | Convolutional VAE | Optional, for a reconstruction vs. likelihood comparison |
| M4 | Multiresolution knowledge distillation (Salehi et al. 2021) | Stretch goal |

This is **one-class classification**. The output is *normal vs. abnormal*, not a disease label. Drusen and CNV never appear in training. They appear only in validation (for an optional informed threshold) and in test (to measure detection).

## 2. Goals and non-goals

**Goals**
- Reproducible results: fixed seeds, splits saved once to disk and reused everywhere.
- No patient leakage between train, val and test.
- Results reported at two levels: **B-scan level** and **patient (case) level**.
- A final comparison table with bootstrap confidence intervals, plus figures for the report and viva.

**Non-goals**
- Multi-class disease classification (Drusen vs. CNV).
- 3D volumetric models. Each B-scan is processed as a 2D image.
- A deployed app or UI.

## 3. Dataset

**Source:** *Labeled Retinal Optical Coherence Tomography Dataset for Classification of Normal, Drusen, and CNV Cases*, Mendeley Data v2 — https://data.mendeley.com/datasets/8kt969dhx6/2
**Authors:** Sotoudeh-Paima, Jodeiri, Hajizadeh, Soltanian-Zadeh (Noor Eye Hospital, Tehran). The dataset page asks you to cite their 2022 multi-scale CNN paper for AMD classification.

| Class folder | Patients | Notes |
|---|---|---|
| NORMAL | 120 | |
| DRUSEN | 160 | |
| CNV | 161 | |
| **Total** | **441** | 16,822 B-scans officially; the local copy has 16,803 |

**Known facts from the earlier inspection (verify in script 01):**
- Folders are `CLASS/<case_number>/...`. **Case numbers restart at 1 in each class**, so `NORMAL/1` and `CNV/1` are different patients.
- Some patient folders have `OD/` and `OS/` subfolders (right and left eye). Others hold images directly.
- The local copy has a mix of `.jpg` and `.tif` files. The Mendeley page lists PNG. Script 01 must report the actual extensions and check whether format correlates with class, since that would be a shortcut a model could learn.
- Each B-scan has its own **specialist label**, encoded in the filename (suffix such as `_normal`, `_drusen`, `_cnv`). It can differ from the folder label:
  - 29 images in the NORMAL folder are labeled Drusen.
  - 2,910 images in the DRUSEN and CNV folders are labeled Normal.
- The dataset ships `data_information.csv` (per-B-scan label, patient and eye). **If it is present, it is the authoritative source for labels and eye.** Filename parsing is the fallback, and the two must be cross-checked.

> **Claude Code:** don't guess the filename pattern. Script 01 must print 20 example paths per class, plus the first rows and columns of `data_information.csv` if it exists. Then confirm the parsing rule with the user before writing `common.py`.

## 4. Key definitions

- `patient_id = f"{CLASS}-{case_number:03d}"`, e.g. `CNV-047`. This is the grouping unit for all splits.
- `folder_label` ∈ {NORMAL, DRUSEN, CNV} is the patient-level diagnosis.
- `bscan_label` ∈ {normal, drusen, cnv} is the specialist label for one image.
- `eye` ∈ {OD, OS, unspecified}.
- **Positive class = abnormal** everywhere: metrics, confusion matrices and curves.
- **Clean-normal pool:** images with `folder_label == NORMAL` **and** `bscan_label == normal`. This is the only data that M1–M4 are trained on in the main experiment.

## 5. Splits (script 03)

Split by patient only. Shuffle the unique patient IDs of each class with `numpy.random.default_rng(42)`, then slice to get these exact counts:

| Class | Train | Val | Test | Ablation-train* |
|---|---|---|---|---|
| NORMAL (120) | 84 | 18 | 18 | 0 |
| DRUSEN (160) | 0 | 40 | 88 | 32 |
| CNV (161) | 0 | 40 | 89 | 32 |

\*`ablation_train` patients are not used in the main experiment at all. They exist only so the expanded-normal-pool ablation (§8) can add normal-labeled slices from diseased eyes without touching val or test. **This differs from the original protocol (30/70 val/test with no reserve).** If the user prefers the original split (DRUSEN 48/112, CNV 48/113) and wants to drop that ablation, set `reserve_ablation_patients: false` in the config.

- Write `outputs/splits.csv` with columns `patient_id, class_folder, split`.
- Every later script **loads** this file. If it already exists, script 03 must refuse to overwrite it unless `--force` is passed.
- Assertions: no `patient_id` appears in more than one split, the counts match the table, and the train split contains only NORMAL patients.

**What each split is used for**

| Split | Contents used | Purpose |
|---|---|---|
| train | Clean-normal B-scans of NORMAL train patients | Fit model weights |
| val (normal) | Clean-normal B-scans of NORMAL val patients | Early stopping; **primary threshold** = 95th percentile of scores |
| val (abnormal) | All B-scans of DRUSEN and CNV val patients | Only for the *informed* threshold (Youden's J), reported alongside the primary one, never instead of it |
| test | All B-scans of all test patients | Touched exactly once, in script 10 |

## 6. Repository layout

```
oct-anomaly/
├── PRD.md
├── config.yaml
├── requirements.txt
├── common.py
├── 01_dataset_check.py
├── 02_build_manifest.py
├── 03_split_data.py
├── 04_preprocess.py
├── 05_train_autoencoder.py
├── 06_evaluate_autoencoder.py
├── 07_train_deep_svdd.py
├── 08_evaluate_deep_svdd.py
├── 09_train_vae.py            # trains and evaluates M3
├── 10_compare_models.py
├── 11_visualize_results.py
├── 12_train_mkd.py            # stretch goal, M4
├── models/                    # saved weights (*.pt)
└── outputs/
    ├── 01_dataset_report.txt
    ├── manifest.csv
    ├── splits.csv
    ├── cache/                 # preprocessed arrays
    ├── scores/                # <method>_scores.csv
    ├── metrics/
    └── figures/
```

**`config.yaml`** (all paths and hyperparameters live here, nothing is hard-coded):

```yaml
data_root: "/path/to/NEH_UT_2021RetinalOCTDataset"   # user sets this
output_dir: "outputs"
seed: 42
image_size: 224
reserve_ablation_patients: true
threshold_percentile: 95
case_aggregation: "max"          # also report "mean" and "top5_mean" in the ablation
bootstrap_iterations: 1000
device: "auto"                   # cuda > mps > cpu
```

**`requirements.txt`:** torch, torchvision, numpy, pandas, scikit-learn, scikit-image, Pillow, matplotlib, pyyaml, tqdm. Python 3.10 or newer.

## 7. Script specifications

Each script: reads `config.yaml`, sets all seeds (`random`, `numpy`, `torch`, with `cudnn.deterministic=True`), logs to the console and to `outputs/logs/<script>.log`, and fails loudly with a clear message if an input file is missing.

### 01_dataset_check.py — read-only inspection
- Walk `data_root` recursively. Import nothing from `common.py`.
- Report:
  - patient and image counts per class
  - folder structure per patient (OD/OS vs. flat)
  - file extensions per class
  - image sizes and modes for a sample of 50 per class
  - `bscan_label` counts crossed with class folder
  - whether `data_information.csv` exists, with its columns and row count
  - unreadable or duplicate files
- Print 20 example paths per class.
- Save everything to `outputs/01_dataset_report.txt`.
- **Accept when:** totals are 120/160/161 patients and about 16.8k images, and the label cross-tab reproduces the 29 and 2,910 counts (or explains any difference).

### common.py — shared helpers (written after 01 is confirmed)
- `load_config()`, `set_seed(seed)`, `get_device()`
- `parse_path(path) -> dict(patient_id, class_folder, case_number, eye, bscan_label)` using the confirmed rule
- `load_manifest()`, `load_splits()`, and `get_split_frame(split, pool="clean_normal"|"all")`
- `OCTDataset(torch.utils.data.Dataset)` that reads from the preprocessed cache, with optional augmentation
- `aggregate_case_scores(df, how="max")`

### 02_build_manifest.py
- One row per image: `image_path, patient_id, class_folder, case_number, eye, bscan_label, ext, height, width`.
- If `data_information.csv` exists, merge it in and write a mismatch report comparing it with the filename labels.
- Output: `outputs/manifest.csv`. **Accept when:** row count equals the file count from 01 and there are no null `patient_id` or `bscan_label` values.

### 03_split_data.py
- Implements §5 exactly. Output: `outputs/splits.csv`, plus per-split image counts printed for both the clean-normal pool and all images.

### 04_preprocess.py
- Load each image as grayscale in float32 [0, 1].
- **Retina ROI crop:** smooth the row-wise mean intensity, find the bright retinal band, crop a fixed-height window around it with a margin, and fall back to a centre crop if detection fails. Log the fallback rate.
- Resize to 224×224 and store as uint8 in `outputs/cache/images.npy` (memmap), with `outputs/cache/index.csv` mapping each row to `image_path`.
- Save a grid of 16 before/after examples to `outputs/figures/preprocess_examples.png` so the crop can be checked by eye.
- Normalization is done at load time: models trained from scratch use [0, 1]; pretrained encoders replicate to 3 channels and apply ImageNet mean and std.
- **Training augmentation** (train split only): horizontal flip p=0.5, rotation ±5°, brightness and contrast ±10%. No augmentation on val or test.

### 05_train_autoencoder.py — M1
- **Architecture:** encoder of 5 blocks (Conv 3×3 → BatchNorm → LeakyReLU → stride-2), with channels 32, 64, 128, 256, 256, going from 224 to 7×7. Bottleneck is a 1×1 conv to 64 channels. The decoder mirrors it with ConvTranspose and ends in a sigmoid.
- **Loss:** MSE. Optional variant: 0.5·MSE + 0.5·(1−SSIM).
- **Training:** Adam, learning rate 1e-3, batch size 32, up to 100 epochs, early stopping with patience 10 on the val-normal reconstruction loss. Save the best checkpoint to `models/ae.pt` and the loss curve to `outputs/figures/ae_loss.png`.

### 06_evaluate_autoencoder.py
- Score **val and test** B-scans. Score = mean squared error per pixel; also save a `1−SSIM` score column.
- Write `outputs/scores/ae_scores.csv` with columns `image_path, patient_id, split, class_folder, bscan_label, score, score_ssim`.
- Save residual heatmaps for 8 test images per class to `outputs/figures/ae_heatmaps/`.
- Do **not** compute test metrics here. That happens in script 10. It is fine to print val metrics.

### 07_train_deep_svdd.py — M2 (primary)
Implement two variants and save both:

1. **Frozen (`svdd_frozen`):** ImageNet ResNet18. Global-average-pool the layer3 and layer4 outputs and concatenate them (256+512 = 768 dimensions), then L2-normalize. Centre `c` = mean of the train embeddings. Score = ‖z − c‖². There is no training loop, so it is a fast and strong baseline.
2. **Fine-tuned (`svdd_ft`):** ResNet18 backbone plus a bias-free linear head to 128 dimensions. Freeze everything up to and including layer2. Compute `c` from an initial pass over train; if any coordinate of `c` has |c_i| < 0.01, set it to ±0.01, following Ruff et al. 2018. Loss = mean ‖z − c‖². Adam, learning rate 1e-4, weight decay 1e-6, 50 epochs, early stopping on the val-normal mean distance. Use no bias terms in the head, to prevent the collapse where every input maps to `c`.

Save `models/svdd_ft.pt` and `models/svdd_centers.pt`.

### 08_evaluate_deep_svdd.py
- Scores go to `outputs/scores/svdd_frozen_scores.csv` and `svdd_ft_scores.csv`, using the same columns as the autoencoder.
- **Heatmaps:** take the layer3 feature map (14×14×256), compute each location's distance to the mean of normal train features at that location, upsample to 224 and overlay. Save them to `outputs/figures/svdd_heatmaps/`.

### 09_train_vae.py — M3 (optional)
- Same encoder and decoder as M1, with a 256-dimensional latent (fc layers from the 7×7 map). Loss = reconstruction (MSE) + β·KL with β=1 and a KL warm-up over the first 10 epochs. Same training protocol as M1.
- Score = reconstruction error; also save a column for reconstruction + KL. Output: `outputs/scores/vae_scores.csv`.

### 10_compare_models.py — the only script that reads test metrics
For every `*_scores.csv`:
1. **Primary threshold** = the `threshold_percentile` of val-normal scores.
2. **Informed threshold** = maximize Youden's J on val (normal plus abnormal). Report it alongside the primary one only.
3. **Test metrics at B-scan level** (ground truth = `bscan_label != normal`) and **case level** (ground truth = `class_folder != NORMAL`, score = `aggregate_case_scores`):
   - ROC-AUC, **PR-AUC**, sensitivity, specificity, precision, F1
   - **Sensitivity for Drusen and CNV separately**
   - confusion matrices at both thresholds
4. **Bootstrap 95% CIs:** 1,000 resamples **by patient**, never by image, for ROC-AUC, PR-AUC and sensitivity.
5. Outputs:
   - `outputs/metrics/comparison_table.csv` and `.md`, one row per method × level
   - `outputs/metrics/<method>_confusion_<level>.csv`

### 11_visualize_results.py
- ROC and PR curves, one overlay per level with every method on it
- Score-distribution histograms for Normal vs. Drusen vs. CNV per method (the most persuasive single figure)
- Confusion-matrix heatmaps
- A heatmap panel for the best method: 3 Normal, 3 Drusen and 3 CNV test images
- All figures go to `outputs/figures/` as PNG at 200 dpi.

### 12_train_mkd.py — M4 (stretch)
- Adapt the official MKD code (Salehi et al., CVPR 2021). The teacher is ImageNet VGG-16, frozen. The student is a smaller VGG-style network that matches the teacher's activations at 4 critical layers. Loss = L_val (Euclidean) + λ·L_dir (cosine). The anomaly score is that same loss at test time.
- It uses the same splits, cache and score-CSV format, so script 10 picks it up automatically.

## 8. Ablations (each one is a flag or small script, reported in a separate table)

| Ablation | What changes | Why it matters |
|---|---|---|
| Clean vs. expanded normal pool | Train M1 and M2 on clean-normal plus normal-labeled slices from `ablation_train` DRUSEN and CNV patients | Tests whether more heterogeneous "normal" data helps or contaminates |
| Naive image-level split | Randomly split images, ignoring patients, then rerun M1 and M2 | Shows as a number how much patient leakage inflates the metrics |
| Case aggregation | max vs. mean vs. top-5 mean | How to turn slice scores into a verdict per eye |
| Threshold | 95th percentile vs. Youden's J vs. oracle (best possible on test, clearly labeled as such) | How sensitive the results are to where the cut-off sits |
| AE score | MSE vs. 1−SSIM | Pixel error vs. structural error |

## 9. Build order and milestones

| Step | Scripts | Done when |
|---|---|---|
| 1 | 01 | Report reviewed; filename rule confirmed with the user |
| 2 | common.py, 02, 03 | manifest and splits pass all assertions |
| 3 | 04 | ROI crop checked visually in the example grid |
| 4 | 05, 06 | AE trained; val ROC-AUC printed |
| 5 | 07, 08 | Both SVDD variants scored; val ROC-AUC printed |
| **Mid-review checkpoint** | | **Validation-set results for M1 and M2**, reported as preliminary. Test stays untouched. |
| 6 | 09 | VAE scored |
| 7 | 10, 11 | Final table and figures |
| 8 | ablations, 12 | Only if time allows |

For the mid-review presentation, the rubric items map as follows:
- **Project, inputs and outputs:** §1
- **Dataset:** §3 and §5
- **Literature survey and model:** Aresta 2025, Salehi 2021, Zhou 2023 → M1–M4
- **Preliminary results:** step 5, on validation only
- **Further plan:** steps 6–8

## 10. Guardrails for Claude Code

1. **Never compute test metrics before script 10.** Scripts 05–09 may print validation metrics only.
2. **Never re-split.** Always load `outputs/splits.csv`.
3. **Train only on the clean-normal pool of train patients,** except in the explicitly flagged ablation.
4. **Resample by patient** in every bootstrap.
5. Do not change hyperparameters in this PRD without saying so and logging the change in `outputs/logs/changes.md`.
6. Build one script at a time. Run it, show the output, and stop for review.
7. If something in the data contradicts this PRD (counts, filenames, labels), stop and report it instead of working around it.

## 11. Limitations to state in the report

- Nothing in the folder structure alone rules out one real person contributing an eye to two different class folders. The dataset design makes this unlikely.
- Single site and single device, so there is no external validation.
- Each B-scan is scored independently, with no 3D context.
- The anomaly score flags abnormality, not a specific disease, so this is a screening aid rather than a diagnosis.

## 12. References

1. Sotoudeh-Paima S. et al. *Labeled Retinal OCT Dataset for Classification of Normal, Drusen, and CNV Cases.* Mendeley Data v2, 2023. https://data.mendeley.com/datasets/8kt969dhx6/2
2. Aresta G. et al. *Anomaly Detection in Retinal OCT Images With Deep Learning-Based Knowledge Distillation.* TVST 14(3):26, 2025. https://doi.org/10.1167/tvst.14.3.26
3. Salehi M. et al. *Multiresolution Knowledge Distillation for Anomaly Detection.* CVPR 2021. https://openaccess.thecvf.com/content/CVPR2021/papers/Salehi_Multiresolution_Knowledge_Distillation_for_Anomaly_Detection_CVPR_2021_paper.pdf
4. Zhou X. et al. *Spatial-contextual variational autoencoder with attention correction for anomaly detection in retinal OCT images.* Computers in Biology and Medicine 152:106328, 2023. https://www.sciencedirect.com/science/article/abs/pii/S0010482522010368
5. Ruff L. et al. *Deep One-Class Classification.* ICML 2018.
