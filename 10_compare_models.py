"""10_compare_models.py - the only script that computes test metrics (PRD section 7).

For every outputs/scores/<method>_scores.csv (column `score`):
  1. primary threshold  = `threshold_percentile` of val-normal scores
                          (B-scan: val clean-normal B-scans; case: val NORMAL patients' case scores)
  2. informed threshold = maximise Youden's J on val (normal + abnormal), reported alongside only
  3. test metrics at B-scan level (truth: bscan_label != normal) and case level (truth: class_folder !=
     NORMAL, score = aggregate_case_scores): ROC-AUC, PR-AUC, sensitivity, specificity, precision, F1,
     sensitivity for Drusen and CNV separately, confusion matrices at both thresholds
  4. bootstrap 95% CIs over test *patients* (resampled with replacement within each class folder) for
     ROC-AUC, PR-AUC and sensitivity at the primary threshold

The excluded layout-outlier patients never enter these metrics; they get their own table.
Ablations that need no retraining (case aggregation, threshold choice, alternative score columns,
file format) go to outputs/metrics/ablations.md.

Outputs: outputs/metrics/comparison_table.{csv,md}, outputs/metrics/<method>_confusion_<level>.csv,
outputs/metrics/excluded_patients.csv, outputs/metrics/ablations.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from common import aggregate_case_scores, load_config, set_seed, setup_logging

LEVELS = ("bscan", "case")
ORDER = ["ae", "ae_mse_ssim", "svdd_frozen", "svdd_ft", "vae", "mkd"]


# ------------------------------------------------------------ helpers
def youden_threshold(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    k = np.argmax(tpr - fpr)
    return float(thr[k])


def binary_metrics(y, s, thr):
    pred = s >= thr
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    return dict(sensitivity=sens, specificity=spec, precision=prec, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def level_frames(df, cfg, how=None):
    """{"bscan": frame, "case": frame} with columns y (bool), score, patient_id, class_folder, sub (drusen/cnv/normal)."""
    how = how or cfg["case_aggregation"]
    b = df.assign(y=df.bscan_label != "normal", sub=df.bscan_label)
    c = aggregate_case_scores(df, how)
    c = c.assign(y=c.class_folder != "NORMAL", sub=c.class_folder.str.lower())
    return {"bscan": b, "case": c}


def thresholds(val, cfg):
    """Primary and informed thresholds per level, from validation data only."""
    p = cfg["threshold_percentile"]
    lv = level_frames(val, cfg)
    clean = val[(val.class_folder == "NORMAL") & (val.bscan_label == "normal")]
    return {
        "bscan": dict(primary=float(np.percentile(clean.score, p)),
                      informed=youden_threshold(lv["bscan"].y.values, lv["bscan"].score.values)),
        "case": dict(primary=float(np.percentile(lv["case"][~lv["case"].y].score, p)),
                     informed=youden_threshold(lv["case"].y.values, lv["case"].score.values)),
    }


def evaluate(frame, thr):
    y, s = frame.y.values, frame.score.values
    row = dict(n=len(frame), n_abnormal=int(y.sum()), roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s))
    for name, t in thr.items():
        m = binary_metrics(y, s, t)
        row.update({f"{k}_{name}": v for k, v in m.items() if k not in ("tp", "fp", "fn", "tn")})
        for sub in ("drusen", "cnv"):
            sel = frame["sub"].values == sub
            row[f"sens_{sub}_{name}"] = float((s[sel] >= t).mean()) if sel.any() else np.nan
        row[f"threshold_{name}"] = t
    return row


def confusion(frame, thr):
    y, s = frame.y.values, frame.score.values
    rows = []
    for name, t in thr.items():
        m = binary_metrics(y, s, t)
        rows.append(dict(threshold=name, value=t, TP=m["tp"], FN=m["fn"], FP=m["fp"], TN=m["tn"]))
    return pd.DataFrame(rows)


def bootstrap(test, cfg, thr, rng, n_iter):
    """95% CIs, resampling test patients with replacement within each class folder."""
    pats = test.groupby("class_folder").patient_id.unique()
    by_pat = {p: g for p, g in test.groupby("patient_id")}
    case_all = level_frames(test, cfg)["case"].set_index("patient_id")
    out = {lv: {k: [] for k in ("roc_auc", "pr_auc", "sensitivity")} for lv in LEVELS}
    for _ in range(n_iter):
        chosen = np.concatenate([rng.choice(p, len(p), replace=True) for p in pats.values])
        b = pd.concat([by_pat[p] for p in chosen], ignore_index=True)
        c = case_all.loc[chosen]
        for lv, fr, y in (("bscan", b, b.bscan_label.values != "normal"), ("case", c, c.y.values)):
            s = fr.score.values
            out[lv]["roc_auc"].append(roc_auc_score(y, s))
            out[lv]["pr_auc"].append(average_precision_score(y, s))
            out[lv]["sensitivity"].append((s[y] >= thr[lv]["primary"]).mean())
    return {lv: {k: np.percentile(v, [2.5, 97.5]) for k, v in d.items()} for lv, d in out.items()}


def md(df, fmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    log = setup_logging("10_compare_models", cfg)
    out = Path(cfg["output_dir"])
    met = out / "metrics"
    met.mkdir(parents=True, exist_ok=True)
    files = sorted((out / "scores").glob("*_scores.csv"))
    if not files:
        raise SystemExit("ERROR: no outputs/scores/*_scores.csv. Run scripts 06, 08, 09 first.")
    methods = sorted((f.name[:-len("_scores.csv")] for f in files),
                     key=lambda m: (ORDER.index(m) if m in ORDER else len(ORDER), m))
    log.info(f"methods: {methods}")
    rng = np.random.default_rng(cfg["seed"])

    rows, excl_rows, abl_agg, abl_thr, abl_score, abl_fmt = [], [], [], [], [], []
    for method in methods:
        df = pd.read_csv(out / "scores" / f"{method}_scores.csv")
        val, test, excl = df[df.split == "val"], df[df.split == "test"], df[df.split == "excluded"]
        thr = thresholds(val, cfg)
        lv = level_frames(test, cfg)
        ci = bootstrap(test, cfg, thr, rng, cfg["bootstrap_iterations"])
        for level in LEVELS:
            r = dict(method=method, level=level, **evaluate(lv[level], thr[level]))
            for k in ("roc_auc", "pr_auc"):
                r[f"{k}_ci"] = f"[{ci[level][k][0]:.3f}, {ci[level][k][1]:.3f}]"
            r["sensitivity_primary_ci"] = f"[{ci[level]['sensitivity'][0]:.3f}, {ci[level]['sensitivity'][1]:.3f}]"
            rows.append(r)
            confusion(lv[level], thr[level]).to_csv(met / f"{method}_confusion_{level}.csv", index=False)
            log.info(f"{method:12s} {level:5s}  ROC-AUC {r['roc_auc']:.3f} {r['roc_auc_ci']}  PR-AUC {r['pr_auc']:.3f} "
                     f"{r['pr_auc_ci']}  sens {r['sensitivity_primary']:.3f}  spec {r['specificity_primary']:.3f}  "
                     f"F1 {r['f1_primary']:.3f}")

        # excluded layout outliers: how many would be flagged, and how extreme are they
        if len(excl):
            excl_rows.append(dict(method=method, images=len(excl),
                                  flagged_at_primary=float((excl.score >= thr["bscan"]["primary"]).mean()),
                                  median_score_percentile_vs_test=float(
                                      (test.score.values[None] <= excl.score.values[:, None]).mean(1).mean() * 100)))

        # ablation: case aggregation
        for how in ("max", "mean", "top5_mean"):
            c = level_frames(test, cfg, how)["case"]
            abl_agg.append(dict(method=method, aggregation=how, roc_auc=roc_auc_score(c.y, c.score),
                                pr_auc=average_precision_score(c.y, c.score)))
        # ablation: threshold choice (B-scan level); oracle = best Youden J on test, for reference only
        b = lv["bscan"]
        oracle = youden_threshold(b.y.values, b.score.values)
        for name, t in (("primary (val p95)", thr["bscan"]["primary"]), ("informed (val Youden)", thr["bscan"]["informed"]),
                        ("ORACLE (test Youden)", oracle)):
            abl_thr.append(dict(method=method, threshold=name, value=t, **{k: v for k, v in binary_metrics(
                b.y.values, b.score.values, t).items() if k in ("sensitivity", "specificity", "f1")}))
        # ablation: alternative score columns (AE 1-SSIM, VAE recon+KL)
        for col in [c for c in df.columns if c.startswith("score")]:
            t2 = test.assign(score=test[col])
            f2 = level_frames(t2, cfg)
            abl_score.append(dict(method=method, score_column=col,
                                  bscan_roc_auc=roc_auc_score(f2["bscan"].y, f2["bscan"].score),
                                  case_roc_auc=roc_auc_score(f2["case"].y, f2["case"].score)))
        # ablation: file format (does the score behave differently for JPG and TIF?)
        for ext, g in test.groupby("ext"):
            y = g.bscan_label != "normal"
            nrm = g[(g.class_folder == "NORMAL") & (g.bscan_label == "normal")]
            abl_fmt.append(dict(method=method, ext=ext, images=len(g), bscan_roc_auc=roc_auc_score(y, g.score),
                                specificity_primary=float((nrm.score < thr["bscan"]["primary"]).mean())))

    tab = pd.DataFrame(rows)
    tab.to_csv(met / "comparison_table.csv", index=False)
    show = tab[["method", "level", "n", "roc_auc", "roc_auc_ci", "pr_auc", "pr_auc_ci", "sensitivity_primary",
                "sensitivity_primary_ci", "specificity_primary", "precision_primary", "f1_primary",
                "sens_drusen_primary", "sens_cnv_primary", "sensitivity_informed", "specificity_informed",
                "f1_informed"]]
    text = ["# Comparison table (test set)", "",
            f"Primary threshold = {cfg['threshold_percentile']}th percentile of val-normal scores; informed threshold "
            "= Youden's J on val. Case score = "
            f"`{cfg['case_aggregation']}` over the patient's B-scans. 95% CIs: {cfg['bootstrap_iterations']} bootstrap "
            "resamples of test patients (within class folder). The excluded layout-outlier patients are not included.",
            "", md(show)]
    (met / "comparison_table.md").write_text("\n".join(text), encoding="utf-8")

    abl = ["# Ablations without retraining (test set)", "",
           "## Case aggregation", "", md(pd.DataFrame(abl_agg)), "",
           "## Threshold choice (B-scan level)", "",
           "The ORACLE row uses test labels and is shown only as an upper reference.", "", md(pd.DataFrame(abl_thr)), "",
           "## Score column", "", md(pd.DataFrame(abl_score)), "",
           "## File format (B-scan level)", "", md(pd.DataFrame(abl_fmt)), ""]
    if excl_rows:
        e = pd.DataFrame(excl_rows)
        e.to_csv(met / "excluded_patients.csv", index=False)
        abl += ["## Excluded layout-outlier patients (never in the main table)", "",
                "Share of their B-scans above the primary B-scan threshold, and their mean percentile among test "
                "scores. High values show the layout shortcut the exclusion avoids.", "", md(e), ""]
    (met / "ablations.md").write_text("\n".join(abl), encoding="utf-8")
    log.info("\n" + "\n".join(text))
    log.info("\n" + "\n".join(abl))


if __name__ == "__main__":
    main()
