"""03_split_data.py - patient-level train / val / test / ablation_train split (PRD section 5).

Per class, the unique patient ids (sorted, minus `exclude_patients`) are shuffled
with numpy.random.default_rng(seed) and sliced in the order train, val, test,
ablation_train using the counts in config.yaml. Excluded patients are written
with split "excluded" so they can be scored separately.

Output: outputs/splits.csv (patient_id, class_folder, split). Refuses to
overwrite an existing file unless --force is given.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import CLASSES, SPLITS, load_config, load_manifest, set_seed, setup_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite an existing outputs/splits.csv")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    log = setup_logging("03_split_data", cfg)
    out = Path(cfg["output_dir"]) / "splits.csv"
    if out.exists() and not args.force:
        sys.exit(f"ERROR: {out} already exists. Splits are made once and reused; pass --force to overwrite.")

    m = load_manifest(cfg)
    counts = cfg["split_counts"] if cfg["reserve_ablation_patients"] else cfg["split_counts_no_reserve"]
    excluded = set(cfg.get("exclude_patients", []))

    rows = []
    for cls in CLASSES:
        pats = sorted(set(m.loc[m.class_folder == cls, "patient_id"]) - excluded)
        want = counts[cls]
        if sum(want[s] for s in SPLITS) != len(pats):
            sys.exit(f"ERROR: split_counts for {cls} sum to {sum(want.values())}, but {len(pats)} patients remain "
                     f"after exclusions. Fix config.yaml.")
        order = np.random.default_rng(cfg["seed"]).permutation(pats)
        start = 0
        for s in SPLITS:
            rows += [dict(patient_id=p, class_folder=cls, split=s) for p in order[start:start + want[s]]]
            start += want[s]
        rows += [dict(patient_id=p, class_folder=cls, split="excluded")
                 for p in sorted(set(m.loc[m.class_folder == cls, "patient_id"]) & excluded)]
    sp = pd.DataFrame(rows)

    # ----------------------------------------------------- assertions
    assert sp.patient_id.is_unique, "a patient appears in more than one split"
    assert set(sp.patient_id) == set(m.patient_id), "splits do not cover every patient in the manifest"
    for cls in CLASSES:
        for s in SPLITS:
            got = ((sp.class_folder == cls) & (sp.split == s)).sum()
            assert got == counts[cls][s], f"{cls}/{s}: {got} patients, expected {counts[cls][s]}"
    assert set(sp.loc[sp.split == "train", "class_folder"]) == {"NORMAL"}, "train contains non-NORMAL patients"

    sp.to_csv(out, index=False)

    # -------------------------------------------------------- report
    log.info("patients per class and split:")
    log.info(pd.crosstab(sp.class_folder, sp.split).reindex(index=CLASSES, columns=SPLITS + ["excluded"], fill_value=0)
             .to_string())
    j = m.merge(sp[["patient_id", "split"]], on="patient_id")
    clean = j[(j.class_folder == "NORMAL") & (j.bscan_label == "normal")]
    log.info("\nclean-normal B-scans per split:")
    log.info(clean.split.value_counts().reindex(SPLITS + ["excluded"], fill_value=0).to_string())
    log.info("\nall B-scans per split (by B-scan label):")
    log.info(pd.crosstab(j.split, j.bscan_label, margins=True).reindex(SPLITS + ["excluded", "All"]).to_string())
    dup = j[j.duplicate_of.fillna("") != ""].drop_duplicates("patient_id")[["patient_id", "split", "duplicate_of"]]
    other = dict(zip(sp.patient_id, sp.split))
    log.info("\nduplicate folders (kept per decision) and where their copies landed:")
    for p, s, d in dup.itertuples(index=False):
        if p > d:
            continue
        log.info(f"  {p} ({s})  <->  {d} ({other[d]})" + ("   SAME SPLIT" if s == other[d] else "   CROSS-SPLIT"))
    log.info(f"\n[PASS] no patient in two splits; counts match config; train is NORMAL only\nsplits -> {out}")


if __name__ == "__main__":
    main()
