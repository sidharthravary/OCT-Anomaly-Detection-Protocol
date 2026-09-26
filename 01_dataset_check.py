"""01_dataset_check.py - read-only inspection of the NEH retinal OCT dataset.

Reads `data_root` from config.yaml, which may be either the extracted dataset
folder or the Mendeley download zip (the image zip nested inside it is read in
place, sequentially, without extracting anything to disk).

Reports what is actually there: patient / image counts, folder structure,
extensions, image sizes and modes, filename labels vs. class folder,
data_information.csv (and how it lines up with the files), unreadable and
duplicate files. Imports nothing from common.py (PRD section 7).

Output: outputs/01_dataset_report.txt and outputs/logs/01_dataset_check.log
"""
import hashlib
import io
import logging
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

CLASSES = ["NORMAL", "DRUSEN", "CNV"]
EXPECTED_PATIENTS = {"NORMAL": 120, "DRUSEN": 160, "CNV": 161}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
EYE_DIRS = {"OD", "OS"}
LABEL_WORDS = ["normal", "drusen", "cnv"]
EXAMPLES_PER_CLASS = 20
INFO_CSV = "data_information.csv"


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


class DataSource:
    """Uniform read access to a dataset folder or a (nested) dataset zip."""

    def __init__(self, root):
        self.root = Path(root)
        if self.root.is_dir():
            self.kind = "dir"
            self.entries = [p.relative_to(self.root).as_posix() + ("/" if p.is_dir() else "")
                            for p in self.root.rglob("*")]
        elif self.root.is_file() and self.root.suffix.lower() == ".zip":
            self.kind = "zip"
            self.outer = zipfile.ZipFile(self.root)
            inner = [n for n in self.outer.namelist() if n.lower().endswith(".zip")]
            if len(inner) > 1:
                sys.exit(f"ERROR: more than one nested zip in {self.root}: {inner}")
            self.inner_name = inner[0] if inner else None
            self.zf = zipfile.ZipFile(self.outer.open(self.inner_name)) if inner else self.outer
            self.offsets = {i.filename: i.header_offset for i in self.zf.infolist()}
            self.entries = [i.filename for i in self.zf.infolist()]
        else:
            sys.exit(f"ERROR: data_root '{self.root}' is neither a folder nor a .zip file.")

    def describe(self):
        if self.kind == "dir":
            return f"folder {self.root.resolve()}"
        return f"zip {self.root.resolve()}" + (f"  ->  nested {self.inner_name}" if self.inner_name else "")

    def iter_bytes(self, paths):
        """Yield (path, bytes). Zip members are read in archive order so the
        nested stream is decompressed once, front to back."""
        if self.kind == "dir":
            for p in paths:
                yield p, (self.root / p).read_bytes()
        else:
            for p in sorted(paths, key=self.offsets.__getitem__):
                yield p, self.zf.read(p)

    def metadata_files(self):
        """Non-image files (csv, py, txt, ...) anywhere in the source, as (name, reader)."""
        out = []
        if self.kind == "dir":
            for p in self.root.rglob("*"):
                if p.is_file() and p.suffix.lower() not in IMAGE_EXTS:
                    out.append((p.relative_to(self.root).as_posix(), p.read_bytes))
        else:
            for n in self.outer.namelist():
                if not n.endswith("/") and n != self.inner_name:
                    out.append((n, lambda n=n: self.outer.read(n)))
            if self.inner_name:
                for n in self.zf.namelist():
                    if not n.endswith("/") and PurePosixPath(n).suffix.lower() not in IMAGE_EXTS:
                        out.append((f"{self.inner_name}!{n}", lambda n=n: self.zf.read(n)))
        return out


def find_class_prefix(entries):
    """Path prefix (tuple of parts) of the folder that holds NORMAL/DRUSEN/CNV."""
    prefixes = Counter()
    for e in entries:
        parts = PurePosixPath(e).parts
        for i, p in enumerate(parts):
            if p.upper() in CLASSES:
                prefixes[parts[:i]] += 1
                break
    return prefixes.most_common(1)[0][0] if prefixes else None


def filename_label(stem):
    """Label word found as a token in the file stem (case-insensitive)."""
    tokens = re.split(r"[_\-\s.()]+", stem.lower())
    found = [w for w in LABEL_WORDS if w in tokens]
    if len(found) == 1:
        return found[0]
    return "<none>" if not found else "<multiple:" + "+".join(found) + ">"


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

    src = DataSource(cfg["data_root"])
    prefix = find_class_prefix(src.entries)
    if prefix is None:
        sys.exit(f"ERROR: no {CLASSES} folders found in {src.describe()}.")
    n_pre = len(prefix)

    emit("=" * 78)
    emit("01 DATASET CHECK")
    emit("=" * 78)
    emit(f"source    : {src.describe()}")
    emit(f"class root: '{'/'.join(prefix) or '.'}'")

    # ------------------------------------------------------------------ index
    # rel = path relative to class root: CLASS/<case>/[OD|OS/]<file>
    images, non_images, case_dirs = [], Counter(), defaultdict(set)
    case_children = defaultdict(lambda: {"subdirs": set(), "direct_imgs": 0})
    for e in src.entries:
        parts = PurePosixPath(e).parts
        if parts[:n_pre] != prefix or len(parts) <= n_pre:
            continue
        rel = parts[n_pre:]
        cls = rel[0].upper()
        if cls not in CLASSES:
            if not e.endswith("/"):
                non_images[f"(outside class folders) {rel[0]}"] += 1
            continue
        if len(rel) >= 2:
            case_dirs[cls].add(rel[1])
        if e.endswith("/"):
            if len(rel) == 3:
                case_children[(cls, rel[1])]["subdirs"].add(rel[2])
            continue
        suffix = PurePosixPath(e).suffix.lower()
        if suffix not in IMAGE_EXTS:
            non_images[suffix or "<none>"] += 1
            continue
        if len(rel) < 3:
            non_images[f"(image directly in {cls})"] += 1
            continue
        if len(rel) == 3:
            case_children[(cls, rel[1])]["direct_imgs"] += 1
        else:
            case_children[(cls, rel[1])]["subdirs"].add(rel[2])
        eye = rel[2].upper() if len(rel) == 4 and rel[2].upper() in EYE_DIRS else "unspecified"
        images.append(dict(entry=e, rel="/".join(rel), cls=cls, case=rel[1], eye_dir=eye, ext=suffix,
                           stem=PurePosixPath(e).stem))

    # ----------------------------------------------------- counts per class
    emit("\n## 1. Patients (case folders) and images per class")
    rows, total_p, total_i, per_patient = [], 0, 0, {}
    for cls in CLASSES:
        cnt = Counter(im["case"] for im in images if im["cls"] == cls)
        per_patient[cls] = np.array(list(cnt.values()))
        n_i = sum(cnt.values())
        rows.append([cls, len(case_dirs[cls]), len(cnt), EXPECTED_PATIENTS[cls], n_i, f"{n_i / max(len(cnt), 1):.1f}"])
        total_p += len(cnt)
        total_i += n_i
    rows.append(["TOTAL", "", total_p, sum(EXPECTED_PATIENTS.values()), total_i, ""])
    emit(table(rows, ["class", "case_dirs", "cases_w_images", "expected", "images", "imgs/case"]))
    for cls in CLASSES:
        v = per_patient[cls]
        emit(f"{cls:7s} images/case: min={v.min()} median={np.median(v):.0f} mean={v.mean():.1f} max={v.max()}")
        bad = sorted(c for c in case_dirs[cls] if not c.isdigit())
        ids = sorted(int(c) for c in case_dirs[cls] if c.isdigit())
        emit(f"        case numbers {ids[0]}..{ids[-1]}, non-numeric: {bad or 'none'}, "
             f"gaps: {sorted(set(range(ids[0], ids[-1] + 1)) - set(ids)) or 'none'}")
    emit("Case numbers restart at 1 in every class folder -> patient_id must include the class.")
    emit(f"Non-image files under class folders: {dict(non_images) or 'none'}")

    # --------------------------------------------------------- structure
    emit("\n## 2. Folder structure per case")
    for cls in CLASSES:
        kinds = Counter()
        for c in case_dirs[cls]:
            ch = case_children[(cls, c)]
            sub = {s.upper() for s in ch["subdirs"]}
            if sub and sub <= EYE_DIRS and not ch["direct_imgs"]:
                kinds["OD/OS: " + "+".join(sorted(sub))] += 1
            elif not sub and ch["direct_imgs"]:
                kinds["flat"] += 1
            else:
                kinds[f"other (subdirs={sorted(ch['subdirs'])}, direct_imgs={ch['direct_imgs']})"] += 1
        emit(f"{cls}: {dict(kinds)}")
    eye_ct = Counter((im["cls"], im["eye_dir"]) for im in images)
    emit(table([[c] + [eye_ct[(c, e)] for e in ["OD", "OS", "unspecified"]] for c in CLASSES],
               ["class", "OD imgs", "OS imgs", "no eye folder"]))

    # --------------------------------------------------------- extensions
    emit("\n## 3. File extensions per class (format-shortcut check)")
    exts = sorted({im["ext"] for im in images})
    ext_ct = Counter((im["cls"], im["ext"]) for im in images)
    emit(table([[c] + [ext_ct[(c, e)] for e in exts] for c in CLASSES], ["class"] + exts))
    for e in exts:
        n = sum(ext_ct[(c, e)] for c in CLASSES)
        emit(f"  {e}: " + ", ".join(f"{c} {100 * ext_ct[(c, e)] / n:.1f}%" for c in CLASSES))
    pat_exts = defaultdict(set)
    for im in images:
        pat_exts[(im["cls"], im["case"])].add(im["ext"])
    emit(f"cases with >1 extension: {sum(1 for v in pat_exts.values() if len(v) > 1)} of {len(pat_exts)}")
    for cls in CLASSES:
        emit(f"  {cls} cases by extension set: "
             f"{dict(Counter('+'.join(sorted(v)) for (c, _), v in pat_exts.items() if c == cls))}")

    # ------------------------------------------------ filename labels
    emit("\n## 4. B-scan label from filename vs. class folder")
    pattern = re.compile(r"^\d+_[A-Za-z]+$")
    emit(f"stems matching '<digits>_<Label>': {sum(bool(pattern.match(im['stem'])) for im in images)} / {len(images)}")
    emit(f"label spellings seen: {dict(Counter(im['stem'].split('_', 1)[-1] for im in images))}")
    for im in images:
        im["fname_label"] = filename_label(im["stem"])
    labels = sorted({im["fname_label"] for im in images})
    lab_ct = Counter((im["cls"], im["fname_label"]) for im in images)
    emit(table([[c] + [lab_ct[(c, l)] for l in labels] + [sum(lab_ct[(c, l)] for l in labels)] for c in CLASSES],
               ["folder \\ fname_label"] + labels + ["total"]))
    n29 = sum(v for (c, l), v in lab_ct.items() if c == "NORMAL" and l != "normal")
    n2910 = lab_ct[("DRUSEN", "normal")] + lab_ct[("CNV", "normal")]
    emit(f"NORMAL folder, non-normal filename label: {n29} (PRD expects 29 drusen)")
    emit(f"DRUSEN+CNV folders, 'normal' filename label: {n2910} (PRD expects 2,910)")
    idx_ct = Counter((im["cls"], im["case"], im["eye_dir"], im["stem"].split("_")[0]) for im in images)
    dup_idx = [k for k, v in idx_ct.items() if v > 1]
    emit(f"(class, case, eye, B-scan index) keys used by >1 file: {len(dup_idx)}"
         + (f" -> cases {sorted({f'{k[0]}/{k[1]}' for k in dup_idx})}" if dup_idx else ""))

    # ------------------------------------------------ data_information.csv
    emit(f"\n## 5. Metadata files and {INFO_CSV}")
    meta = src.metadata_files()
    for name, _ in meta:
        emit(f"found: {name}")
    info = [(n, r) for n, r in meta if PurePosixPath(n.split("!")[-1]).name.lower() == INFO_CSV]
    if not info:
        emit(f"{INFO_CSV} NOT present -> filename parsing will be the label source.")
    else:
        df = pd.read_csv(io.BytesIO(info[0][1]()))
        emit(f"{INFO_CSV}: {len(df)} rows, columns = {list(df.columns)}")
        emit(df.head(8).to_string())
        for col in df.columns:
            if df[col].nunique() <= 20:
                emit(f"  '{col}': {df[col].value_counts(dropna=False).to_dict()}")
        emit("Patient ID range per Class: " + ", ".join(
            f"{c} {g.min()}..{g.max()} ({g.nunique()} ids)" for c, g in df.groupby("Class")["Patient ID"]))
        disk = {im["rel"] for im in images}
        dup_rows = df[df.duplicated(keep="first")]
        dirs = set(df["Directory"])
        emit(f"exact duplicate rows: {len(dup_rows)}")
        emit(f"distinct Directory values: {len(dirs)}")
        missing = sorted(dirs - disk)
        emit(f"Directory values with no file: {len(missing)} {missing[:10]}")
        low = Counter(d.lower() for d in dirs)
        emit(f"  ... of which differ from another row only by letter case: "
             f"{sum(1 for d in missing if low[d.lower()] > 1)} (lost on a case-insensitive filesystem)")
        emit(f"files with no Directory row: {len(disk - dirs)} {sorted(disk - dirs)[:10]}")
        emit(f"explained: {len(df)} rows - {len(dup_rows)} duplicates - {len(missing)} missing = "
             f"{len(df) - len(dup_rows) - len(missing)} vs {len(images)} files")
        d = df.drop_duplicates("Directory").set_index("Directory")
        on_disk = [im for im in images if im["rel"] in d.index]
        lab_mis = sum(d.at[im["rel"], "Label"].lower() != im["fname_label"] for im in on_disk)
        eye_mis = sum(im["eye_dir"] != "unspecified" and d.at[im["rel"], "Eye"] != im["eye_dir"] for im in on_disk)
        emit(f"CSV Label vs filename label disagreements: {lab_mis}")
        emit(f"CSV Eye vs OD/OS folder disagreements: {eye_mis} "
             f"(CSV gives an eye for all {len(on_disk)} files, including flat folders)")

    # ------------------------------------------ full pass over all images
    emit("\n## 6. Image size, mode, readability and duplicates (every image, one sequential pass)")
    by_entry = {im["entry"]: im for im in images}
    unreadable, hashes = [], defaultdict(list)
    for entry, data in tqdm(src.iter_bytes(list(by_entry)), total=len(by_entry), desc="reading", file=sys.stdout,
                            mininterval=5):
        im = by_entry[entry]
        hashes[hashlib.md5(data).hexdigest()].append(im)
        try:
            with Image.open(io.BytesIO(data)) as img:
                im["size"], im["mode"] = img.size, img.mode
                img.load()
        except Exception as e:
            unreadable.append((im["rel"], f"{type(e).__name__}: {e}"))
            im["size"], im["mode"] = "ERR", "ERR"
    size_ct = Counter((im["cls"], im["ext"], im["size"], im["mode"]) for im in images)
    emit(table([[c, e, f"{s[0]}x{s[1]}" if s != "ERR" else s, m, n] for (c, e, s, m), n in sorted(size_ct.items(), key=str)],
               ["class", "ext", "WxH", "mode", "images"]))
    emit(f"unreadable: {len(unreadable)}")
    for rel, err in unreadable[:50]:
        emit(f"  {rel} -> {err}")
    dup_groups = [g for g in hashes.values() if len(g) > 1]
    emit(f"byte-identical duplicate groups: {len(dup_groups)} ({sum(len(g) for g in dup_groups)} files)")
    cross_patient = [g for g in dup_groups if len({(im['cls'], im['case']) for im in g}) > 1]
    cross_class = [g for g in dup_groups if len({im["cls"] for im in g}) > 1]
    cross_label = [g for g in dup_groups if len({im["fname_label"] for im in g}) > 1]
    emit(f"  spanning >1 case: {len(cross_patient)}   >1 class: {len(cross_class)}   >1 label: {len(cross_label)}")
    for g in dup_groups[:25]:
        emit("  " + " | ".join(im["rel"] for im in g))

    # ------------------------------------------------------ example paths
    emit(f"\n## 7. {EXAMPLES_PER_CLASS} example paths per class (relative to class root)")
    rng = random.Random(cfg["seed"])
    for cls in CLASSES:
        pool = sorted(im["rel"] for im in images if im["cls"] == cls)
        emit(f"-- {cls}")
        for rel in sorted(rng.sample(pool, min(EXAMPLES_PER_CLASS, len(pool)))):
            emit(f"   {rel}")

    # ---------------------------------------------------- acceptance
    emit("\n## 8. Acceptance checks")
    ok_p = all(len(per_patient[c]) == EXPECTED_PATIENTS[c] for c in CLASSES)
    emit(f"[{'PASS' if ok_p else 'FAIL'}] cases per class = 120/160/161 -> "
         + "/".join(str(len(per_patient[c])) for c in CLASSES))
    emit(f"[{'PASS' if 16_000 <= total_i <= 17_500 else 'FAIL'}] ~16.8k images -> {total_i}")
    emit(f"[{'PASS' if n29 == 29 else 'CHECK'}] NORMAL-folder abnormal labels = {n29} (expect 29)")
    emit(f"[{'PASS' if n2910 == 2910 else 'CHECK'}] DRUSEN/CNV-folder normal labels = {n2910} (expect 2,910)")
    emit(f"[{'PASS' if not unreadable else 'FAIL'}] unreadable files = {len(unreadable)}")
    emit(f"[{'PASS' if not cross_patient else 'CHECK'}] duplicate images across cases = {len(cross_patient)}")

    (out_dir / "01_dataset_report.txt").write_text("\n".join(report), encoding="utf-8")
    log.info(f"\nReport written to {out_dir / '01_dataset_report.txt'}")


if __name__ == "__main__":
    main()
