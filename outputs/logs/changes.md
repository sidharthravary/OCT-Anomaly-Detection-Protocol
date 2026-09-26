# Changes relative to PRD.md

Every deviation from the PRD is logged here (PRD section 10, guardrail 5).

## 2026-09-26 - data handling (steps 1-2)

1. **`data_root` may point at the Mendeley zip.** The nested image zip is read in place, in archive order,
   because the machine has too little free disk to extract 4.2 GB. Results are identical to reading an
   extracted folder; `DataSource` in `common.py` supports both.
2. **7 layout-outlier patients excluded from the main experiment:** CNV-159, CNV-160, CNV-161, DRUSEN-157,
   DRUSEN-158, DRUSEN-159, DRUSEN-160 (208 B-scans). Their scans are exported as a small picture in the
   top-left of a black canvas with a text box (EDA section 4); none is NORMAL, so layout alone would give
   them away. They get split `excluded` in `splits.csv` and are to be scored and reported separately.
   Decided by the user.
3. **Split counts changed as a consequence** (PRD section 5 table): DRUSEN test 88 -> 84, CNV test 89 -> 86.
   Train, val and ablation_train counts are unchanged. The no-reserve alternative is 47/109 (DRUSEN) and
   47/111 (CNV). Counts live in `config.yaml` (`split_counts`, `split_counts_no_reserve`).
4. **Duplicate patient folders kept as-is** (user decision "keep all, ignore"): CNV-022 = DRUSEN-069,
   CNV-026 = CNV-147, DRUSEN-017 = NORMAL-034, DRUSEN-022 = DRUSEN-083 hold byte-identical images. The
   manifest records them in `duplicate_of`. With seed 42 they land in: test/val, val/ablation_train,
   ablation_train/train, test/test. To be stated as a limitation in the report.
5. **Seeding detail:** each class is shuffled with its own `default_rng(seed)`, so one class's split does
   not depend on the order in which classes are processed.

## 2026-09-26 - preprocessing (step 3)

6. **New config key `roi_height: 352`.** PRD section 7 asks for "a fixed-height window around the retina" without a
   number. On a 1-in-25 sample, the detected retina (Otsu mask, 2nd-98th percentile of per-column top/bottom)
   fits with an 8 px margin in 82.5% / 90.4% / 95.8% / 98.3% / 99.4% of images for heights 256 / 288 / 320 / 352 /
   384. 352 rows x full width (768) is cropped, then resized to 224 x 224. Same height for every image, so the
   crop scale carries no class information. Full run: 0% fallback; retina extends past the window in 0.4% of
   NORMAL, 0.4% of DRUSEN and 1.7% of CNV images.
7. **Scale bar blacked out** (rows 455-491, columns 0-85) before cropping, in every image.

## 2026-09-27 - models and evaluation (steps 4-7)

8. **Hardware: training on CPU** (PyTorch 2.14 CPU build). The CUDA build did not fit on the disk (needs ~8 GB
   free while installing). Same code runs on GPU when `device: auto` finds CUDA; results can differ slightly.
9. **`nets.py` added** (not in the PRD layout): holds ConvAutoencoder, ConvVAE, ResNet18 feature extractor,
   DeepSVDD and SSIM, so that training and scoring scripts share one definition.
10. **AE decoder detail:** ConvTranspose 4x4 stride 2 blocks mirror the encoder; the 1x1 bottleneck (64 ch) is
    mapped back to 256 ch by a 1x1 conv + LeakyReLU before the decoder.
11. **svdd_ft BatchNorm:** BN layers stay in eval mode (ImageNet running statistics) with frozen affine
    parameters. Their shifts act as bias terms, which Deep SVDD must avoid (Ruff et al. 2018). Early-stopping
    patience for svdd_ft = 10 (PRD gives none).
12. **VAE loss scale:** reconstruction term = squared error *summed* over pixels (not averaged), so it is on the
    same scale as the KL summed over the 256 latent dims; with a per-pixel mean the KL term would dominate
    and the VAE would collapse to the mean image. Early stopping uses the val loss at beta = 1 and only counts
    from the end of the 10-epoch warm-up. `score` = per-pixel MSE (comparable to M1), `score_kl` = SSE + KL.
13. **Excluded patients are scored** by 06/08/09 (split `excluded`) and reported in a separate table by 10.
14. **Case-level primary threshold** = 95th percentile of val NORMAL patients' case scores (18 patients);
    B-scan-level primary threshold = 95th percentile of val clean-normal B-scan scores.
15. **Bootstrap** resamples test patients with replacement *within each class folder*, so every resample has
    both normal and abnormal cases.
16. **Optional `cache_dir` config key** (default `outputs/cache`) so the 843 MB cache can live outside OneDrive.
