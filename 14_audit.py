"""14_audit.py - end-to-end check-up of data, splits, scores, saved models and evaluation.

Prints one PASS / WARN / FAIL line per check and a summary. Read-only except for nothing: it writes no files.
Run it after script 10 (it re-computes test metrics only to verify comparison_table.csv).

  A  data and splits      manifest, splits, leakage, cache
  B  score files          completeness, NaN/inf, labels
  C  saved models         reload every model and re-score random val B-scans; must match the score CSVs
  D  model behaviour      does each model separate normal from abnormal, no collapse, threshold calibration
  E  evaluation           independent re-computation of test metrics, CIs, confusion matrices, guardrails
  F  repository           git clean, README numbers match the table
"""
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from common import (CLASSES, SPLITS, OCTDataset, aggregate_case_scores, cache_dir, get_device, get_split_frame,
                    load_config, load_manifest, load_splits, scoring_frame, set_seed)

RESULTS = []


def check(name, ok, detail="", warn=False):
    """ok -> PASS; not ok -> WARN if warn else FAIL."""
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    RESULTS.append(status)
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))


# ------------------------------------------------------------------ A
def grade(name, value, fail_below, warn_below, detail):
    """FAIL below fail_below, WARN below warn_below, else PASS."""
    if value < fail_below:
        check(name, False, detail)
    else:
        check(name, value >= warn_below, detail, warn=True)


def audit_data(cfg):
    print("\n== A. data and splits")
    m, s = load_manifest(cfg), load_splits(cfg)
    check("manifest: 16,803 rows, unique image_path, no null ids/labels",
          len(m) == 16803 and m.image_path.is_unique and m[["patient_id", "bscan_label"]].notna().all().all(),
          f"{len(m)} rows")
    check("splits: every manifest patient exactly once", s.patient_id.is_unique and set(s.patient_id) == set(m.patient_id),
          f"{s.patient_id.nunique()} patients")
    counts = cfg["split_counts"] if cfg["reserve_ablation_patients"] else cfg["split_counts_no_reserve"]
    ok = all(((s.class_folder == c) & (s.split == k)).sum() == counts[c][k] for c in CLASSES for k in SPLITS)
    check("split counts match config.yaml", ok)
    check("train split holds NORMAL patients only", set(s[s.split == "train"].class_folder) == {"NORMAL"})
    sets = {k: set(s[s.split == k].patient_id) for k in SPLITS + ["excluded"]}
    overlap = [(a, b) for i, a in enumerate(sets) for b in list(sets)[i + 1:] if sets[a] & sets[b]]
    check("no patient in two splits (train/val/test/ablation_train/excluded)", not overlap, str(overlap or ""))
    check("excluded patients = exclude_patients in config", sets["excluded"] == set(cfg["exclude_patients"]))
    tr = get_split_frame(cfg, "train")
    check("training pool is clean-normal only (NORMAL folder + normal label)",
          set(tr.class_folder) == {"NORMAL"} and set(tr.bscan_label) == {"normal"}, f"{len(tr)} B-scans")
    idx = pd.read_csv(cache_dir(cfg) / "index.csv")
    X = np.load(cache_dir(cfg) / "images.npy", mmap_mode="r")
    check("cache aligned with manifest, uint8 N x 224 x 224",
          (idx.image_path.values == m.image_path.values).all() and X.shape == (len(m), 224, 224) and X.dtype == np.uint8)
    means = X.reshape(len(X), -1)[:, ::97].mean(1)
    check("no blank / saturated cached image", (means > 1).all() and (means < 250).all(),
          f"subsampled means {means.min():.1f}..{means.max():.1f}")
    check("ROI detection fallback rate", idx.fallback.mean() < 0.01, f"{idx.fallback.mean():.2%}")


# ------------------------------------------------------------------ B
def audit_scores(cfg):
    print("\n== B. score files")
    ref = scoring_frame(cfg)
    files = sorted((Path(cfg["output_dir"]) / "scores").glob("*_scores.csv"))
    out = {}
    for f in files:
        d = pd.read_csv(f)
        name = f.name[:-len("_scores.csv")]
        out[name] = d
        cols = [c for c in d.columns if c.startswith("score")]
        ok = (len(d) == len(ref) and set(d.image_path) == set(ref.image_path) and d.image_path.is_unique
              and np.isfinite(d[cols].values).all())
        same = d.set_index("image_path").loc[ref.image_path]
        ok_lab = (same.bscan_label.values == ref.bscan_label.values).all() and (same.split.values == ref.split.values).all()
        check(f"{name}: {len(d)} rows = val+test+excluded, finite scores {cols}, labels/splits match manifest",
              ok and ok_lab, f"{d.split.value_counts().to_dict()}")
    return out


# ------------------------------------------------------------------ C / D
@torch.no_grad()
def rescore(cfg, name, frame, device):
    """Re-compute scores for `frame` with the saved model, exactly as the scoring scripts do."""
    from nets import (ConvAutoencoder, ConvVAE, DeepSVDD, ResNetFeatures, VGGStudent, VGGTeacher, frozen_embedding,
                      kl_divergence, mkd_loss)
    norm = "unit" if name in ("ae", "vae") else "imagenet"
    dl = DataLoader(OCTDataset(cfg, frame, norm=norm), batch_size=32, num_workers=0)
    extra = {}
    if name == "ae":
        net = ConvAutoencoder().to(device)
        net.load_state_dict(torch.load("models/ae.pt", map_location=device)["state_dict"])
        f = lambda x: ((net(x) - x) ** 2).flatten(1).mean(1)
    elif name == "vae":
        net = ConvVAE().to(device)
        net.load_state_dict(torch.load("models/vae.pt", map_location=device)["state_dict"])
        extra["kl"] = []

        def f(x):
            x_hat, mu, lv = net(x)
            extra["kl"].append(kl_divergence(mu, lv).cpu())
            return ((x_hat - x) ** 2).flatten(1).mean(1)
    elif name == "svdd_frozen":
        net = ResNetFeatures().to(device)
        c = torch.load("models/svdd_centers.pt", map_location=device)["frozen_c"]
        f = lambda x: ((frozen_embedding(*net(x)) - c) ** 2).sum(1)
    elif name == "svdd_ft":
        ck = torch.load("models/svdd_ft.pt", map_location=device)
        net = DeepSVDD(ck["hp"]["rep_dim"]).to(device)
        net.load_state_dict(ck["state_dict"])
        c = torch.load("models/svdd_centers.pt", map_location=device)["ft_c"]
        extra["z"] = []

        def f(x):
            z = net(x)
            extra["z"].append(z.cpu())
            return ((z - c) ** 2).sum(1)
    elif name == "mkd":
        net = VGGStudent().to(device)
        net.load_state_dict(torch.load("models/mkd.pt", map_location=device)["state_dict"])
        teacher = VGGTeacher().to(device)

        def f(x):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                fs, ft = net(x), teacher(x)
            return mkd_loss(fs, ft, 0.01)
    else:
        return None, {}
    net.eval()
    s = torch.cat([f(x.to(device)).float().cpu() for x, _ in dl]).numpy()
    return s, extra


def audit_models(cfg, scores, device):
    print(f"\n== C. saved models reload and reproduce their scores (device {device})")
    rng = np.random.default_rng(0)
    for name, d in scores.items():
        v = d[d.split == "val"]
        pick = v.iloc[rng.choice(len(v), 96, replace=False)].reset_index(drop=True)
        s, extra = rescore(cfg, name, pick, device)
        if s is None:
            check(f"{name}: no reload rule", False, warn=True)
            continue
        # difference relative to the spread of the scores: robust for methods whose scores are tiny
        # (svdd_ft distances ~3e-4 are differences of nearly equal float32 numbers)
        rel = np.abs(s - pick.score.values) / d[d.split == "val"].score.std()
        tol = 2e-2 if name == "mkd" else 1e-2        # MKD scores are computed under bfloat16 autocast
        check(f"{name}: re-scored 96 val B-scans match the CSV", rel.max() < tol,
              f"max |diff| = {rel.max():.1e} x score std")
        r = np.corrcoef(s, pick.score.values)[0, 1]
        check(f"{name}: ranking identical (Pearson r of re-score vs CSV)", r > 0.999, f"r = {r:.5f}")
        if name == "svdd_ft":
            z = torch.cat(extra["z"]).numpy()
            spread = z.std(0).mean()
            c = torch.load("models/svdd_centers.pt", map_location="cpu")["ft_c"].numpy()
            ratio = spread / np.abs(c).mean()
            grade("svdd_ft: embedding spread vs centre size (collapse check)", ratio, 1e-4, 0.05,
                  f"per-dim std {spread:.2e}, mean |c| {np.abs(c).mean():.2e}, ratio {ratio:.4f}")
        if name == "vae":
            kl = torch.cat(extra["kl"]).numpy()
            check("vae: no posterior collapse (mean KL per B-scan > 1 nat)", kl.mean() > 1, f"mean KL {kl.mean():.1f}")

    print("\n== D. model behaviour on validation (test not used here)")
    p = cfg["threshold_percentile"]
    for name, d in scores.items():
        v = d[d.split == "val"]
        clean = v[(v.class_folder == "NORMAL") & (v.bscan_label == "normal")]
        abn = v[v.bscan_label != "normal"]
        auc = roc_auc_score(v.bscan_label != "normal", v.score)
        grade(f"{name}: val ROC-AUC above chance", auc, 0.55, 0.6, f"{auc:.3f}")
        med = (abn.score.median(), clean.score.median())
        check(f"{name}: median score abnormal > clean-normal", med[0] > med[1], f"{med[0]:.4g} vs {med[1]:.4g}")
        cnv = v[v.bscan_label.isin(["normal", "cnv"])]
        a_cnv = roc_auc_score(cnv.bscan_label == "cnv", cnv.score)
        grade(f"{name}: detects CNV (normal vs CNV ROC-AUC)", a_cnv, 0.65, 0.75, f"{a_cnv:.3f}")
        thr = np.percentile(clean.score, p)
        spec = (clean.score < thr).mean()
        check(f"{name}: primary threshold gives ~{p}% specificity on val clean-normal", abs(spec - p / 100) < 0.01,
              f"{spec:.3f}")
        t = d[d.split == "test"]
        tc = t[(t.class_folder == "NORMAL") & (t.bscan_label == "normal")]
        spec_t = (tc.score < thr).mean()
        check(f"{name}: threshold transfers to test clean-normal (specificity within 5 points)",
              abs(spec_t - spec) < 0.05, f"test {spec_t:.3f} vs val {spec:.3f}", warn=True)


# ------------------------------------------------------------------ E
def audit_evaluation(cfg, scores):
    print("\n== E. evaluation integrity")
    met = Path(cfg["output_dir"]) / "metrics"
    tab = pd.read_csv(met / "comparison_table.csv")
    for name, d in scores.items():
        t = d[d.split == "test"]
        roc_b = roc_auc_score(t.bscan_label != "normal", t.score)
        ap_b = average_precision_score(t.bscan_label != "normal", t.score)
        c = aggregate_case_scores(t, cfg["case_aggregation"])
        roc_c = roc_auc_score(c.class_folder != "NORMAL", c.score)
        r = tab[tab.method == name].set_index("level")
        ok = (abs(r.at["bscan", "roc_auc"] - roc_b) < 1e-9 and abs(r.at["bscan", "pr_auc"] - ap_b) < 1e-9
              and abs(r.at["case", "roc_auc"] - roc_c) < 1e-9)
        check(f"{name}: comparison_table ROC/PR-AUC re-computed independently", ok,
              f"B-scan {roc_b:.3f}, case {roc_c:.3f}")
        for lvl in ("bscan", "case"):
            lo, hi = map(float, re.findall(r"[\d.]+", r.at[lvl, "roc_auc_ci"]))
            check(f"{name} {lvl}: 95% CI contains the point estimate", lo <= r.at[lvl, "roc_auc"] <= hi, f"[{lo}, {hi}]")
            cm = pd.read_csv(met / f"{name}_confusion_{lvl}.csv")
            n = r.at[lvl, "n"]
            check(f"{name} {lvl}: confusion matrices sum to n = {n}",
                  (cm[["TP", "FN", "FP", "TN"]].sum(axis=1) == n).all())
    check("case level: 188 test patients (18 + 84 + 86)", (tab[tab.level == "case"].n == 188).all())

    # static guardrails on the code
    src = {p.name: p.read_text(encoding="utf-8") for p in Path(".").glob("[01][0-9]*.py") if p.name != "14_audit.py"}
    metric_calls = re.compile(r"roc_auc_score|average_precision_score|val_summary")
    leaks = []
    for fname, text in src.items():
        if fname.startswith(("10_", "11_", "14_", "01")):
            continue
        for line in text.splitlines():
            if metric_calls.search(line) and "test" in line:
                leaks.append(f"{fname}: {line.strip()[:80]}")
    check("guardrail: no test metrics in scripts 02-09, 12, 13 (only val_summary on val)", not leaks, "; ".join(leaks))
    aug = [(f, l.strip()) for f, t in src.items() for l in t.splitlines() if "augment=True" in l]
    bad = [a for a in aug if "train" not in a[1]]
    check("guardrail: augmentation only on training loaders", not bad, f"{len(aug)} uses, all train" if not bad else str(bad))
    common = Path("common.py").read_text(encoding="utf-8")
    check("guardrail: val_summary filters split == 'val'", 'v = df[df.split == "val"]' in common)


# ------------------------------------------------------------------ F
def audit_repo(cfg):
    print("\n== F. repository")
    st = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    check("git working tree clean (everything committed)", not st, st.replace("\n", "; ")[:200], warn=True)
    tab = pd.read_csv(Path(cfg["output_dir"]) / "metrics" / "comparison_table.csv")
    readme = Path("README.md").read_text(encoding="utf-8")
    missing = [f"{r.method}/{r.level} {r.roc_auc:.3f}" for r in tab.itertuples() if f"{r.roc_auc:.3f}" not in readme]
    check("README reports the same ROC-AUCs as comparison_table.csv", not missing, ", ".join(missing))


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    audit_data(cfg)
    scores = audit_scores(cfg)
    audit_models(cfg, scores, device)
    audit_evaluation(cfg, scores)
    audit_repo(cfg)
    print(f"\nSUMMARY: {RESULTS.count('PASS')} PASS, {RESULTS.count('WARN')} WARN, {RESULTS.count('FAIL')} FAIL")


if __name__ == "__main__":
    main()
