"""01_dataset_check.py - read-only inspection of the NEH retinal OCT dataset.

Walks `data_root` from config.yaml and reports what is actually on disk:
patient / image counts, folder structure, extensions, image sizes and modes,
filename-label candidates vs. class folder, data_information.csv, unreadable
and duplicate files. Imports nothing from common.py (PRD section 7).

Output: outputs/01_dataset_report.txt and outputs/logs/01_dataset_check.log
"""
import hashlib
import logging
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

CLASSES = ["NORMAL", "DRUSEN", "CNV"]
EXPECTED_PATIENTS = {"NORMAL": 120, "DRUSEN": 160, "CNV": 161}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
EYE_DIRS = {"OD", "OS"}
LABEL_WORDS = ["normal", "drusen", "cnv"]
SAMPLE_PER_CLASS = 50
EXAMPLES_PER_CLASS = 20


def load_config(path="config.yaml"):
    if not Path(path).is_file():
        sys.exit(f"ERROR: {path} not found. Run from the project root.")
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def setup_logging(output_dir):
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("01")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(log_dir / "01_dataset_check.log", mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def find_class_root(data_root):
    """Return the directory that directly contains NORMAL/DRUSEN/CNV folders."""
    for dirpath, dirnames, _ in os.walk(data_root):
        upper = {d.upper() for d in dirnames}
        if all(c in upper for c in CLASSES):
            return Path(dirpath)
    return None


def filename_label_candidates(stem):
    """Label words found as tokens in the file stem (case-insensitive)."""
    tokens = re.split(r"[_\-\s.()]+", stem.lower())
    return [w for w in LABEL_WORDS if w in tokens]


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def table(rows, header):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = lambda r: "  ".join(str(x).ljust(w) for x, w in zip(r, widths))
    return "\n".join([line(header), line(["-" * w for w in widths])] + [line(r) for r in rows])


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir)
    report = []

    def emit(msg=""):
        log.info(msg)
        report.append(msg)

    data_root = Path(cfg["data_root"])
    if not data_root.is_dir():
        sys.exit(f"ERROR: data_root '{data_root}' does not exist. Set it in config.yaml.")

    class_root = find_class_root(data_root)
    if class_root is None:
        sys.exit(f"ERROR: no directory under '{data_root}' contains all of {CLASSES}.")
    class_dirs = {d.name.upper(): d for d in class_root.iterdir() if d.is_dir() and d.name.upper() in CLASSES}

    emit("=" * 78)
    emit("01 DATASET CHECK")
    emit("=" * 78)
    emit(f"data_root : {data_root.resolve()}")
    emit(f"class_root: {class_root.resolve()}")
    other_top = [p.name for p in class_root.iterdir() if p.name.upper() not in CLASSES]
    emit(f"other entries next to class folders: {other_top or 'none'}")

    # ------------------------------------------------------------------ walk
    images = []           # dicts: path, cls, case, eye_dir, ext
    non_images = Counter()
    structure = {c: Counter() for c in CLASSES}
    non_numeric_cases = defaultdict(list)
    for cls in CLASSES:
        for case_dir in sorted(p for p in class_dirs[cls].iterdir() if p.is_dir()):
            if not case_dir.name.isdigit():
                non_numeric_cases[cls].append(case_dir.name)
            subdirs = sorted(p.name for p in case_dir.iterdir() if p.is_dir())
            direct_imgs = [p for p in case_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            if subdirs and set(s.upper() for s in subdirs) <= EYE_DIRS and not direct_imgs:
                kind = "OD/OS: " + "+".join(sorted(s.upper() for s in subdirs))
            elif not subdirs and direct_imgs:
                kind = "flat"
            elif subdirs and direct_imgs:
                kind = f"mixed (images + subdirs {subdirs})"
            elif subdirs:
                kind = f"other subdirs {subdirs}"
            else:
                kind = "empty"
            structure[cls][kind] += 1
            for p in sorted(case_dir.rglob("*")):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in IMAGE_EXTS:
                    non_images[p.suffix.lower() or "<none>"] += 1
                    continue
                rel_parts = p.relative_to(case_dir).parts
                eye = rel_parts[0].upper() if len(rel_parts) > 1 and rel_parts[0].upper() in EYE_DIRS else "unspecified"
                images.append(dict(path=p, cls=cls, case=case_dir.name, eye_dir=eye, ext=p.suffix.lower()))
        # stray files directly in the class folder
        for p in class_dirs[cls].iterdir():
            if p.is_file():
                non_images[f"(file directly in {cls}) {p.suffix.lower()}"] += 1

    # ----------------------------------------------------- counts per class
    emit("\n## 1. Patients and images per class")
    rows, total_p, total_i = [], 0, 0
    for cls in CLASSES:
        n_p = len({im["case"] for im in images if im["cls"] == cls})
        n_dirs = sum(1 for p in class_dirs[cls].iterdir() if p.is_dir())
        n_i = sum(1 for im in images if im["cls"] == cls)
        rows.append([cls, n_dirs, n_p, EXPECTED_PATIENTS[cls], n_i, f"{n_i / max(n_p, 1):.1f}"])
        total_p += n_p
        total_i += n_i
    rows.append(["TOTAL", "", total_p, sum(EXPECTED_PATIENTS.values()), total_i, ""])
    emit(table(rows, ["class", "case_dirs", "patients_w_images", "expected", "images", "imgs/patient"]))
    for cls in CLASSES:
        if non_numeric_cases[cls]:
            emit(f"WARNING: non-numeric case folders in {cls}: {non_numeric_cases[cls][:20]}")
    per_patient = defaultdict(list)
    for cls in CLASSES:
        cnt = Counter(im["case"] for im in images if im["cls"] == cls)
        v = np.array(list(cnt.values()))
        per_patient[cls] = v
        emit(f"{cls:7s} images/patient: min={v.min()} median={np.median(v):.0f} mean={v.mean():.1f} max={v.max()}")
    emit(f"Non-image files: {dict(non_images) or 'none'}")

    # --------------------------------------------------------- structure
    emit("\n## 2. Folder structure per patient")
    for cls in CLASSES:
        emit(f"{cls}: {dict(structure[cls])}")
    eye_ct = Counter((im["cls"], im["eye_dir"]) for im in images)
    emit(table([[c] + [eye_ct[(c, e)] for e in ["OD", "OS", "unspecified"]] for c in CLASSES],
               ["class", "OD imgs", "OS imgs", "unspecified imgs"]))

    # --------------------------------------------------------- extensions
    emit("\n## 3. File extensions per class (format shortcut check)")
    exts = sorted({im["ext"] for im in images})
    ext_ct = Counter((im["cls"], im["ext"]) for im in images)
    emit(table([[c] + [ext_ct[(c, e)] for e in exts] for c in CLASSES], ["class"] + exts))
    for e in exts:
        n = sum(ext_ct[(c, e)] for c in CLASSES)
        emit(f"  {e}: " + ", ".join(f"{c} {100 * ext_ct[(c, e)] / n:.1f}%" for c in CLASSES))
    # extension per patient (is format constant within a patient?)
    pat_exts = defaultdict(set)
    for im in images:
        pat_exts[(im["cls"], im["case"])].add(im["ext"])
    mixed = sum(1 for v in pat_exts.values() if len(v) > 1)
    emit(f"patients with >1 extension: {mixed}")

    # ---------------------------------------------- sizes / modes (sample)
    emit(f"\n## 4. Image size and mode (random sample of {SAMPLE_PER_CLASS} per class)")
    rng = random.Random(cfg["seed"])
    for cls in CLASSES:
        pool = [im for im in images if im["cls"] == cls]
        sample = rng.sample(pool, min(SAMPLE_PER_CLASS, len(pool)))
        sizes, modes, by_ext = Counter(), Counter(), Counter()
        for im in sample:
            try:
                with Image.open(im["path"]) as img:
                    sizes[img.size] += 1          # (width, height)
                    modes[img.mode] += 1
                    by_ext[(im["ext"], img.size, img.mode)] += 1
            except Exception as e:  # reported again in section 7
                sizes[f"ERR {type(e).__name__}"] += 1
        emit(f"{cls}: sizes(WxH)={dict(sizes)}  modes={dict(modes)}")
        emit(f"        (ext, size, mode)={dict(by_ext)}")

    # ------------------------------------------------ filename labels
    emit("\n## 5. B-scan label from filename vs. class folder")
    last_tok = Counter()
    for im in images:
        cands = filename_label_candidates(im["path"].stem)
        im["fname_label"] = cands[0] if len(cands) == 1 else ("<none>" if not cands else "<multiple:" + "+".join(cands) + ">")
        last_tok[(im["cls"], re.split(r"[_\-\s.()]+", im["path"].stem.lower())[-1])] += 1
    labels = sorted({im["fname_label"] for im in images})
    lab_ct = Counter((im["cls"], im["fname_label"]) for im in images)
    emit(table([[c] + [lab_ct[(c, l)] for l in labels] + [sum(lab_ct[(c, l)] for l in labels)] for c in CLASSES],
               ["folder \\ fname_label"] + labels + ["total"]))
    emit(f"NORMAL folder, non-normal filename label: {sum(v for (c, l), v in lab_ct.items() if c == 'NORMAL' and l != 'normal')} (PRD expects 29 drusen)")
    emit(f"DRUSEN+CNV folders, 'normal' filename label: {lab_ct[('DRUSEN', 'normal')] + lab_ct[('CNV', 'normal')]} (PRD expects 2,910)")
    emit("Most common last stem token per class (helps confirm the naming rule):")
    for cls in CLASSES:
        top = [(t, n) for (c, t), n in last_tok.most_common() if c == cls][:10]
        emit(f"  {cls}: {top}")

    # ------------------------------------------------ data_information.csv
    emit("\n## 6. data_information.csv")
    info_files = [p for p in data_root.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls", ".txt"}]
    if not info_files:
        emit("No csv/xlsx/txt metadata file found anywhere under data_root.")
    for p in info_files:
        emit(f"found: {p.relative_to(data_root)}  ({p.stat().st_size} bytes)")
    info = [p for p in info_files if p.name.lower() == "data_information.csv"]
    if info:
        import pandas as pd
        df = pd.read_csv(info[0])
        emit(f"data_information.csv: {len(df)} rows, columns = {list(df.columns)}")
        emit(df.head(10).to_string())
        for col in df.columns:
            if df[col].nunique() <= 20:
                emit(f"  value counts of '{col}': {df[col].value_counts(dropna=False).to_dict()}")
        emit(f"row count vs. images on disk: {len(df)} vs {len(images)}")
    else:
        emit("data_information.csv NOT present -> filename parsing will be the label source.")

    # ------------------------------------------ unreadable / duplicates
    emit("\n## 7. Unreadable and duplicate files (all images)")
    unreadable, hashes = [], defaultdict(list)
    for im in images:
        try:
            with Image.open(im["path"]) as img:
                img.verify()
        except Exception as e:
            unreadable.append((im["path"], f"{type(e).__name__}: {e}"))
        hashes[md5(im["path"])].append(im)
    emit(f"unreadable: {len(unreadable)}")
    for p, err in unreadable[:50]:
        emit(f"  {p.relative_to(class_root)} -> {err}")
    dup_groups = [g for g in hashes.values() if len(g) > 1]
    emit(f"byte-identical duplicate groups: {len(dup_groups)} ({sum(len(g) for g in dup_groups)} files)")
    cross_patient = [g for g in dup_groups if len({(im['cls'], im['case']) for im in g}) > 1]
    cross_class = [g for g in dup_groups if len({im["cls"] for im in g}) > 1]
    emit(f"  groups spanning >1 patient: {len(cross_patient)}   spanning >1 class: {len(cross_class)}")
    for g in dup_groups[:20]:
        emit("  " + " | ".join(str(im["path"].relative_to(class_root)) for im in g))

    # ------------------------------------------------------ example paths
    emit(f"\n## 8. {EXAMPLES_PER_CLASS} example paths per class (relative to class_root)")
    for cls in CLASSES:
        pool = [im for im in images if im["cls"] == cls]
        emit(f"-- {cls}")
        for im in rng.sample(pool, min(EXAMPLES_PER_CLASS, len(pool))):
            emit(f"   {im['path'].relative_to(class_root).as_posix()}")

    # ---------------------------------------------------- acceptance
    emit("\n## 9. Acceptance checks")
    ok_p = all(len(per_patient[c]) == EXPECTED_PATIENTS[c] for c in CLASSES)
    emit(f"[{'PASS' if ok_p else 'FAIL'}] patients per class = 120/160/161 -> "
         + "/".join(str(len(per_patient[c])) for c in CLASSES))
    ok_i = 16_000 <= total_i <= 17_500
    emit(f"[{'PASS' if ok_i else 'FAIL'}] ~16.8k images -> {total_i}")
    n29 = sum(v for (c, l), v in lab_ct.items() if c == "NORMAL" and l != "normal")
    n2910 = lab_ct[("DRUSEN", "normal")] + lab_ct[("CNV", "normal")]
    emit(f"[{'PASS' if n29 == 29 else 'CHECK'}] NORMAL-folder abnormal labels = {n29} (expect 29)")
    emit(f"[{'PASS' if n2910 == 2910 else 'CHECK'}] DRUSEN/CNV-folder normal labels = {n2910} (expect 2,910)")
    emit(f"[{'PASS' if not unreadable else 'FAIL'}] unreadable files = {len(unreadable)}")
    emit(f"[{'PASS' if not cross_patient else 'CHECK'}] duplicates across patients = {len(cross_patient)}")

    (out_dir / "01_dataset_report.txt").write_text("\n".join(report), encoding="utf-8")
    log.info(f"\nReport written to {out_dir / '01_dataset_report.txt'}")


if __name__ == "__main__":
    main()
