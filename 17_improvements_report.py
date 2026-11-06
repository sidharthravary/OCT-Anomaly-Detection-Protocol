"""17_improvements_report.py - ensemble + neighbouring-slice smoothing on top of M1-M5, and the final report.

Still one-class: every score comes from models fitted on clean-normal TRAIN B-scans only, and every choice below
is made on NEH VALIDATION, fixed before the test set is read (changes.md item 28):
  1. PatchCore variant  = the pc_* column with the higher val B-scan ROC-AUC.
  2. Ensemble           = mean of per-model z-scores (standardised by val clean-normal mean / std, so no disease
                          labels are needed for the scaling); the member subset (>= 2 of the 5 candidates) with the
                          highest val B-scan ROC-AUC is chosen.
  3. Slice smoothing    = moving mean of a B-scan's score with its neighbours in the same eye (window 1 / 3 / 5);
                          the window with the highest val B-scan ROC-AUC is chosen, per score.
Then test (NEH B-scan + case, worst-case subset vs Sotoudeh-Paima 2022) and Kermany (vs Aresta 2025) are reported
for every candidate, chosen or not. Output: outputs/improvements/results.md (+ .csv).
"""
import importlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import load_config, load_manifest, set_seed, setup_logging

OUT = Path("outputs/improvements")
BASE = ["ae_mapmax", "svdd_frozen", "vae_mapmax", "mkd"]     # val-chosen scores of M1-M4 (15_external_kermany)
ARESTA = {"all": 0.87, "DRUSEN": 0.70, "CNV": 0.95, "DME": 0.95}
SOTOUDEH = dict(acc=92.0, sens=91.8, spec=95.8)              # FPN-VGG16, Table 6 (supervised, 3-class average)


def auc(y, s):
    y = np.asarray(y)
    return roc_auc_score(y, s) if 0 < y.mean() < 1 else np.nan


def zscore(df, ref, cols):
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = (df[c] - ref[c].mean()) / ref[c].std()
    return out


def smooth(df, col, k):
    if k == 1:
        return df[col]
    order = df.sort_values(["patient_id", "eye", "bscan_index"])
    s = order.groupby(["patient_id", "eye"])[col].transform(lambda v: v.rolling(k, center=True, min_periods=1).mean())
    return s.reindex(df.index)


def worst_case(d):
    keep = ((d.class_folder == "NORMAL") & (d.bscan_label == "normal")) | \
           ((d.class_folder == "DRUSEN") & (d.bscan_label == "drusen")) | ((d.class_folder == "CNV") & (d.bscan_label == "cnv"))
    return d[keep]


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    log = setup_logging("17_improvements_report", cfg)
    boot_ci = importlib.import_module("15_external_kermany").patient_boot_ci

    neh = pd.read_csv("outputs/external/neh_scores_with_mapmax.csv")
    pc = pd.read_csv(OUT / "neh_patchcore_scores.csv")[["image_path", "pc_r18_l23", "pc_r18_l12"]]
    neh = neh.merge(pc, on="image_path").merge(load_manifest(cfg)[["image_path", "bscan_index"]], on="image_path")
    ker = pd.read_csv("outputs/external/kermany_scores.csv")
    ker = ker[ker.variant == "harmonised"].merge(
        pd.read_csv(OUT / "kermany_patchcore_scores.csv")[["path", "pc_r18_l23", "pc_r18_l12"]], on="path")
    val, test = neh[neh.split == "val"], neh[neh.split == "test"]
    vnorm = val[(val.class_folder == "NORMAL") & (val.bscan_label == "normal")]
    yv = val.bscan_label != "normal"
    choice = []

    # 1. PatchCore variant
    pv = {c: auc(yv, val[c]) for c in ("pc_r18_l23", "pc_r18_l12")}
    pc_col = max(pv, key=pv.get)
    neh["patchcore"], ker["patchcore"] = neh[pc_col], ker[pc_col]
    choice.append(f"PatchCore variant: {pc_col} (val AUC " + ", ".join(f"{k} {v:.3f}" for k, v in pv.items()) + ")")

    # 2. Ensemble of z-scores (scaled by val clean-normal statistics only)
    cands = BASE + ["patchcore"]
    vnorm = neh.loc[vnorm.index]
    zn, zk = zscore(neh, vnorm, cands), zscore(ker, vnorm, cands)
    val = neh[neh.split == "val"]
    best, best_auc, subset_aucs = None, -1, []
    for r in range(2, len(cands) + 1):
        for sub in combinations(cands, r):
            a = auc(yv, zn.loc[val.index, list(sub)].mean(1))
            subset_aucs.append((sub, a))
            if a > best_auc:
                best, best_auc = sub, a
    neh["ensemble"], ker["ensemble"] = zn[list(best)].mean(1), zk[list(best)].mean(1)
    neh["ensemble_all"], ker["ensemble_all"] = zn[cands].mean(1), zk[cands].mean(1)
    choice.append(f"Ensemble members: {' + '.join(best)} (val AUC {best_auc:.3f}; all five: "
                  f"{auc(yv, neh.loc[val.index, 'ensemble_all']):.3f}; {len(subset_aucs)} subsets tried)")

    # 3. Neighbouring-slice smoothing, window chosen on val per score
    for col in ("patchcore", "ensemble"):
        va = {k: auc(yv, smooth(neh, col, k).loc[val.index]) for k in (1, 3, 5)}
        k = max(va, key=va.get)
        neh[f"{col}_smooth"] = smooth(neh, col, k)
        choice.append(f"Slice smoothing for {col}: window {k} (val AUC " + ", ".join(f"w{w} {a:.3f}" for w, a in va.items()) + ")")

    # ---- evaluation (test read only here) ----
    val, test = neh[neh.split == "val"], neh[neh.split == "test"]
    vnorm = val[(val.class_folder == "NORMAL") & (val.bscan_label == "normal")]
    wc_t = worst_case(test)
    rows = []
    cols = ["svdd_frozen", "vae_mapmax", "ae_mapmax", "mkd", "pc_r18_l23", "pc_r18_l12", "patchcore", "ensemble",
            "ensemble_all", "patchcore_smooth", "ensemble_smooth"]
    for c in cols:
        thr95 = np.percentile(vnorm[c], 95)
        thr_s = np.percentile(vnorm[c], SOTOUDEH["spec"])
        yt = test.bscan_label != "normal"
        nd = test.bscan_label != "cnv"
        ncv = test.bscan_label != "drusen"
        case = test.groupby("patient_id").agg(class_folder=("class_folder", "first"), s=(c, "max")).reset_index()
        lo, hi = boot_ci(case.rename(columns={"s": c}), c, np.random.default_rng(cfg["seed"]))
        ywc = wc_t.class_folder != "NORMAL"
        r = dict(score=c,
                 val_auc=auc(yv, val[c]),
                 bscan_auc=auc(yt, test[c]), bscan_pr=average_precision_score(yt, test[c]),
                 drusen_auc=auc(test[nd].bscan_label == "drusen", test[nd][c]),
                 cnv_auc=auc(test[ncv].bscan_label == "cnv", test[ncv][c]),
                 sens_p95=float((test.loc[yt, c] > thr95).mean()), spec_p95=float((test.loc[~yt, c] <= thr95).mean()),
                 case_auc=auc(case.class_folder != "NORMAL", case.s), case_ci=f"[{lo:.3f}, {hi:.3f}]",
                 wc_auc=auc(ywc, wc_t[c]),
                 wc_sens=float((wc_t.loc[ywc, c] > thr_s).mean()), wc_spec=float((wc_t.loc[~ywc, c] <= thr_s).mean()))
        if c in ker:
            yk = ker.class_folder != "NORMAL"
            r["k_auc"] = auc(yk, ker[c])
            for cls in ("DRUSEN", "CNV", "DME"):
                m = ker.class_folder.isin(["NORMAL", cls])
                r[f"k_{cls.lower()}"] = auc(ker[m].class_folder == cls, ker[m][c])
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results.csv", index=False)
    neh.to_csv(OUT / "neh_all_scores.csv", index=False)

    f = lambda v: "" if pd.isna(v) else f"{v:.3f}"
    lab = {"svdd_frozen": "M2 frozen Deep SVDD (previous best)", "vae_mapmax": "M3 VAE, map-max",
           "ae_mapmax": "M1 AE, map-max", "mkd": "M4 MKD", "pc_r18_l23": "M5 PatchCore, layer2+3",
           "pc_r18_l12": "M5 PatchCore, layer1+2", "patchcore": "M5 PatchCore (val-chosen)",
           "ensemble": "Ensemble (val-chosen members)", "ensemble_all": "Ensemble (all five)",
           "patchcore_smooth": "M5 PatchCore + slice smoothing", "ensemble_smooth": "Ensemble + slice smoothing"}
    L = ["# Improvements: patch-level detection, ensemble, slice smoothing (one-class)", "",
         "All models are fitted on clean-normal train B-scans only (no disease images). Every choice was made on NEH "
         "validation before reading test. Test = 188 patients / 6,716 B-scans; case = max over a patient's B-scans "
         "(95% CI: 1,000 patient bootstraps). Thresholds from val clean-normal scores.", "",
         "## Choices (validation only)", ""] + [f"- {c}" for c in choice] + [
         "", "## NEH test", "",
         "| method | val B-scan AUC | B-scan AUC | PR-AUC (base 0.656) | drusen AUC | CNV AUC | sens / spec @ p95 | case AUC [95% CI] |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append(f"| {lab[r.score]} | {f(r.val_auc)} | {f(r.bscan_auc)} | {f(r.bscan_pr)} | {f(r.drusen_auc)} | "
                 f"{f(r.cnv_auc)} | {f(r.sens_p95)} / {f(r.spec_p95)} | {f(r.case_auc)} {r.case_ci} |")
    L += ["", "## Versus Sotoudeh-Paima et al. 2022 (same worst-case B-scan subset of our test patients)", "",
          f"Supervised FPN-VGG16 (3-class, trained with disease labels, patient-level 5-fold CV): accuracy "
          f"{SOTOUDEH['acc']}%, sensitivity {SOTOUDEH['sens']}%, specificity {SOTOUDEH['spec']}% (3-class averages). "
          "Ours: normal vs abnormal, threshold = 95.8th percentile of val clean-normal scores.", "",
          "| method | worst-case AUC | sensitivity | specificity |", "|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append(f"| {lab[r.score]} | {f(r.wc_auc)} | {f(r.wc_sens)} | {f(r.wc_spec)} |")
    L += ["", "## Versus Aresta et al. 2025 (Kermany test, B-scan AUC; no slice smoothing: single B-scans)", "",
          "| method | all abnormal | drusen | CNV | DME |", "|---|---|---|---|---|",
          f"| Aresta 2025 | {ARESTA['all']} | {ARESTA['DRUSEN']} | {ARESTA['CNV']} | {ARESTA['DME']} |"]
    for _, r in res.iterrows():
        if not pd.isna(r.get("k_auc")):
            L.append(f"| {lab[r.score]} | {f(r.k_auc)} | {f(r.k_drusen)} | {f(r.k_cnv)} | {f(r.k_dme)} |")
    (OUT / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info("\n".join(L))


if __name__ == "__main__":
    main()
