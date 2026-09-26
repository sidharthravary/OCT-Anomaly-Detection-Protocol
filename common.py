"""common.py - shared helpers for scripts 02 onwards (PRD section 7).

Filename rule (confirmed in 01 / 01b): paths below the class root look like
    CLASS/<case_number>/[OD|OS/]<bscan_index>_<Label>.<ext>
Case numbers restart at 1 in every class, so patient_id = f"{CLASS}-{case:03d}".
The label suffix is spelled inconsistently (Normal/NOrmal/normal, Drusen/DRusen)
and is lower-cased. data_information.csv is authoritative for label and eye;
02_build_manifest.py cross-checks it against this rule.
"""
import logging
import random
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import yaml

try:  # torch is only needed from script 04 on
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object

CLASSES = ["NORMAL", "DRUSEN", "CNV"]
LABELS = ["normal", "drusen", "cnv"]
SPLITS = ["train", "val", "test", "ablation_train"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_PATH_RE = re.compile(r"^(?P<cls>[A-Za-z]+)/(?P<case>\d+)/(?:(?P<eye>OD|OS)/)?(?P<idx>\d+)_(?P<label>[A-Za-z]+)\.\w+$",
                      re.IGNORECASE)


# ------------------------------------------------------------------ setup
def load_config(path="config.yaml"):
    if not Path(path).is_file():
        sys.exit(f"ERROR: {path} not found. Run from the project root.")
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(pref="auto"):
    if torch is None:
        sys.exit("ERROR: PyTorch is not installed (pip install -r requirements.txt).")
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(script_name, cfg):
    """Log to the console and to outputs/logs/<script_name>.log."""
    log_dir = Path(cfg["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(log_dir / f"{script_name}.log", mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def require(path, hint):
    """Fail loudly if an input file from an earlier script is missing."""
    if not Path(path).exists():
        sys.exit(f"ERROR: {path} not found. {hint}")
    return Path(path)


# ------------------------------------------------------------ raw data
class DataSource:
    """Read access to the dataset as an extracted folder or the Mendeley zip.

    The zip holds data_information.csv, read_data.py and a nested image zip;
    the nested zip is read in place. `image_paths` are relative to the class
    root (CLASS/case/...), and `iter_bytes` yields them in archive order so the
    nested stream is decompressed once, front to back.
    """

    def __init__(self, data_root):
        root = Path(data_root)
        if root.is_dir():
            self.kind, self.root = "dir", root
            files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
            names = {self._rel(p.relative_to(root).as_posix()): p for p in files}
            self._read = lambda rel: names[rel].read_bytes()
            self._order = lambda rels: rels
            self._info = next(root.rglob("data_information.csv"), None)
            self._info_bytes = self._info.read_bytes() if self._info else None
        elif root.suffix.lower() == ".zip":
            self.kind = "zip"
            outer = zipfile.ZipFile(root)
            inner = [n for n in outer.namelist() if n.lower().endswith(".zip")]
            zf = zipfile.ZipFile(outer.open(inner[0])) if inner else outer
            infos = [i for i in zf.infolist() if PurePosixPath(i.filename).suffix.lower() in IMAGE_EXTS]
            names = {self._rel(i.filename): i for i in infos}
            self._read = lambda rel: zf.read(names[rel])
            self._order = lambda rels: sorted(rels, key=lambda r: names[r].header_offset)
            info = [n for n in outer.namelist() if n.endswith("data_information.csv")]
            self._info_bytes = outer.read(info[0]) if info else None
        else:
            sys.exit(f"ERROR: data_root '{root}' is neither a folder nor a .zip file.")
        self.image_paths = sorted(names)

    @staticmethod
    def _rel(path):
        parts = PurePosixPath(path).parts
        i = next((k for k, p in enumerate(parts) if p.upper() in CLASSES), None)
        if i is None:
            sys.exit(f"ERROR: no class folder in path '{path}'.")
        return "/".join(parts[i:])

    def info_csv_bytes(self):
        return self._info_bytes

    def iter_bytes(self, rels=None):
        for rel in self._order(list(rels if rels is not None else self.image_paths)):
            yield rel, self._read(rel)


def parse_path(path):
    """CLASS/<case>/[OD|OS/]<idx>_<Label>.<ext>  ->  dict of identifiers."""
    m = _PATH_RE.match(str(path).replace("\\", "/"))
    if m is None or m["cls"].upper() not in CLASSES or m["label"].lower() not in LABELS:
        raise ValueError(f"path does not follow the naming rule: {path}")
    cls, case = m["cls"].upper(), int(m["case"])
    return dict(patient_id=f"{cls}-{case:03d}", class_folder=cls, case_number=case,
                eye=m["eye"].upper() if m["eye"] else "unspecified", bscan_index=int(m["idx"]),
                bscan_label=m["label"].lower())


# ------------------------------------------------------ manifest/splits
def load_manifest(cfg):
    p = require(Path(cfg["output_dir"]) / "manifest.csv", "Run 02_build_manifest.py first.")
    return pd.read_csv(p)


def load_splits(cfg):
    p = require(Path(cfg["output_dir"]) / "splits.csv", "Run 03_split_data.py first.")
    return pd.read_csv(p)


def get_split_frame(cfg, split, pool="clean_normal"):
    """Manifest rows of one split.

    pool="clean_normal": NORMAL-folder patients' normal-labelled B-scans only.
    pool="all": every B-scan of the split's patients.
    pool="normal_labelled": every normal-labelled B-scan (any folder), for the expanded-pool ablation.
    """
    m = load_manifest(cfg).merge(load_splits(cfg)[["patient_id", "split"]], on="patient_id", how="inner")
    m = m[m.split == split]
    if pool == "clean_normal":
        m = m[(m.class_folder == "NORMAL") & (m.bscan_label == "normal")]
    elif pool == "normal_labelled":
        m = m[m.bscan_label == "normal"]
    elif pool != "all":
        raise ValueError(f"unknown pool '{pool}'")
    return m.reset_index(drop=True)


# -------------------------------------------------------------- dataset
class OCTDataset(Dataset):
    """B-scans from the preprocessed cache written by 04_preprocess.py.

    norm="unit"     -> 1 x H x W in [0, 1]           (models trained from scratch)
    norm="imagenet" -> 3 x H x W, ImageNet mean/std  (pretrained encoders)
    augment=True applies the PRD training augmentation (train split only).
    Returns (image, index into `frame`).
    """

    def __init__(self, cfg, frame, norm="unit", augment=False):
        if torch is None:
            sys.exit("ERROR: PyTorch is not installed (pip install -r requirements.txt).")
        cache = Path(cfg["output_dir"]) / "cache"
        index = pd.read_csv(require(cache / "index.csv", "Run 04_preprocess.py first."))
        self.images = np.load(require(cache / "images.npy", "Run 04_preprocess.py first."), mmap_mode="r")
        row_of = dict(zip(index.image_path, index.row))
        missing = [p for p in frame.image_path if p not in row_of]
        if missing:
            sys.exit(f"ERROR: {len(missing)} images are not in the cache, e.g. {missing[:3]}. Re-run 04.")
        self.rows = np.array([row_of[p] for p in frame.image_path])
        self.norm = norm
        self.augment = None
        if augment:
            from torchvision import transforms as T
            self.augment = T.Compose([T.RandomHorizontalFlip(0.5), T.RandomRotation(5),
                                      T.ColorJitter(brightness=0.1, contrast=0.1)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x = torch.from_numpy(np.asarray(self.images[self.rows[i]], dtype=np.float32) / 255.0)[None]
        if self.augment is not None:
            x = self.augment(x).clamp(0, 1)
        if self.norm == "imagenet":
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            x = (x.expand(3, -1, -1) - mean) / std
        return x, i


# ------------------------------------------------------------- scoring
def aggregate_case_scores(df, how="max", group_col="patient_id", score_col="score"):
    """Collapse B-scan scores to one score per case (patient by default).

    how: "max" | "mean" | "top5_mean". Keeps class_folder and split per case.
    """
    g = df.groupby(group_col)
    if how == "max":
        s = g[score_col].max()
    elif how == "mean":
        s = g[score_col].mean()
    elif how == "top5_mean":
        s = g[score_col].apply(lambda v: v.nlargest(5).mean())
    else:
        raise ValueError(f"unknown aggregation '{how}'")
    keep = [c for c in ("class_folder", "split") if c in df.columns]
    return g[keep].first().assign(**{score_col: s}).reset_index()
