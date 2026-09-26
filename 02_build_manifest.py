"""02_build_manifest.py - one row per B-scan image.

Columns: image_path, patient_id, class_folder, case_number, eye, bscan_label,
ext, height, width, plus bscan_index, eye_folder, md5, duplicate_of, excluded.

Labels and eye come from data_information.csv when present (authoritative),
cross-checked against the filename rule in common.parse_path. The comparison
is written to outputs/02_mismatch_report.txt.

Output: outputs/manifest.csv
"""
import hashlib
import io
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import DataSource, load_config, parse_path, set_seed, setup_logging


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    log = setup_logging("02_build_manifest", cfg)
    out = Path(cfg["output_dir"])
    src = DataSource(cfg["data_root"])
    log.info(f"{len(src.image_paths)} image files in {cfg['data_root']}")

    rows = []
    for rel in src.image_paths:
        r = parse_path(rel)
        r["eye_folder"] = r.pop("eye")
        rows.append(dict(image_path=rel, ext=Path(rel).suffix.lower(), **r))
    m = pd.DataFrame(rows).rename(columns={"bscan_label": "label_filename"})

    # ---------------------------------------------- data_information.csv
    rep = ["02 MANIFEST - data_information.csv vs. filenames", ""]
    info_bytes = src.info_csv_bytes()
    if info_bytes is None:
        log.info("data_information.csv not found -> labels from filenames, eye from OD/OS folders")
        m["bscan_label"], m["eye"] = m.label_filename, m.eye_folder
        rep.append("data_information.csv not present; filename rule used for every field.")
    else:
        info = pd.read_csv(io.BytesIO(info_bytes))
        dups = info[info.duplicated(keep="first")]
        conflicting = info.drop_duplicates()[info.drop_duplicates().Directory.duplicated(keep=False)]
        info = info.drop_duplicates("Directory")
        missing = sorted(set(info.Directory) - set(m.image_path))
        extra = sorted(set(m.image_path) - set(info.Directory))
        rep += [f"CSV rows: {len(info) + len(dups)}; exact duplicate rows dropped: {len(dups)}",
                f"Directory values with conflicting rows: {len(conflicting)}",
                f"CSV rows whose file is missing: {len(missing)}"]
        rep += [f"  {p}" for p in missing]
        rep += [f"files without a CSV row: {len(extra)}"] + [f"  {p}" for p in extra]
        m = m.merge(info.rename(columns={"Directory": "image_path", "Eye": "eye", "Label": "label_csv",
                                         "Patient ID": "csv_patient", "Class": "csv_class", "B-scan": "csv_bscan"}),
                    on="image_path", how="left")
        has = m.label_csv.notna()
        m["bscan_label"] = m.label_csv.str.lower().where(has, m.label_filename)
        m["eye"] = m.eye.where(has, m.eye_folder)
        checks = {
            "label (CSV vs filename)": has & (m.label_csv.str.lower() != m.label_filename),
            "eye (CSV vs OD/OS folder)": has & (m.eye_folder != "unspecified") & (m.eye != m.eye_folder),
            "patient number (CSV vs folder)": has & (m.csv_patient != m.case_number),
            "class (CSV vs folder)": has & (m.csv_class.str.upper() != m.class_folder),
            "B-scan index (CSV vs filename)": has & (m.csv_bscan != m.bscan_index),
        }
        for name, bad in checks.items():
            rep.append(f"mismatches in {name}: {int(bad.sum())}")
            rep += [f"  {p}" for p in m.image_path[bad].head(20)]
        m = m.drop(columns=["label_csv", "csv_patient", "csv_class", "csv_bscan"])
    m["eye"] = m.eye.fillna("unspecified")

    # ------------------------------------------- size and content hash
    size, md5 = {}, {}
    for rel, data in tqdm(src.iter_bytes(), total=len(m), desc="reading", file=sys.stdout, mininterval=10):
        with Image.open(io.BytesIO(data)) as img:
            size[rel] = img.size
        md5[rel] = hashlib.md5(data).hexdigest()
    m["width"] = m.image_path.map(lambda p: size[p][0])
    m["height"] = m.image_path.map(lambda p: size[p][1])
    m["md5"] = m.image_path.map(md5)
    owners = defaultdict(set)
    for p, h in zip(m.patient_id, m.md5):
        owners[h].add(p)
    m["duplicate_of"] = [";".join(sorted(owners[h] - {p})) for p, h in zip(m.patient_id, m.md5)]

    excl = set(cfg.get("exclude_patients", []))
    unknown = excl - set(m.patient_id)
    if unknown:
        sys.exit(f"ERROR: exclude_patients not in the data: {sorted(unknown)}")
    m["excluded"] = m.patient_id.isin(excl).map({True: "layout_outlier", False: ""})

    cols = ["image_path", "patient_id", "class_folder", "case_number", "eye", "bscan_label", "ext", "height", "width",
            "bscan_index", "eye_folder", "md5", "duplicate_of", "excluded"]
    m = m[cols].sort_values(["class_folder", "case_number", "eye", "bscan_index", "image_path"]).reset_index(drop=True)

    # ----------------------------------------------------- acceptance
    assert len(m) == len(src.image_paths), "row count differs from file count"
    assert m.patient_id.notna().all() and m.bscan_label.notna().all(), "null patient_id or bscan_label"
    assert m.image_path.is_unique, "duplicate image_path"
    assert set(m.bscan_label) <= {"normal", "drusen", "cnv"}, f"unexpected labels {set(m.bscan_label)}"

    m.to_csv(out / "manifest.csv", index=False)
    dup = m[m.duplicate_of != ""]
    rep += ["", f"images sharing identical bytes with another patient: {len(dup)} "
                f"(kept, per decision; see outputs/logs/changes.md)",
            *[f"  {a} <-> {b}" for a, b in sorted({tuple(sorted((p, q))) for p, qs in zip(dup.patient_id, dup.duplicate_of)
                                                   for q in qs.split(";")})]]
    (out / "02_mismatch_report.txt").write_text("\n".join(rep), encoding="utf-8")

    log.info("\n".join(rep))
    log.info("\nimages per class folder x B-scan label:")
    log.info(pd.crosstab(m.class_folder, m.bscan_label, margins=True).to_string())
    log.info(f"\neye: {m.eye.value_counts().to_dict()}")
    log.info(f"excluded (layout outliers): {m.excluded.eq('layout_outlier').sum()} images, "
             f"{m[m.excluded != ''].patient_id.nunique()} patients")
    log.info(f"\n[PASS] {len(m)} rows == {len(src.image_paths)} files; no null patient_id / bscan_label")
    log.info(f"manifest -> {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
