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
