"""13_ablation_retrain.py - PRD section 8 ablations that need retraining.

  --variant expanded  train on clean-normal train B-scans PLUS the normal-labelled B-scans of the
                      ablation_train DRUSEN / CNV patients (more heterogeneous "normal"); val / test unchanged
  --variant naive     ignore patients: randomly split the main-experiment B-scans into train / val / test
                      (same per-class image shares as the patient split), then train and score. Normal
                      slices of the same patient end up in train and test, which shows how much leakage
                      inflates the metrics. The image-level split is saved once to outputs/splits_naive.csv.
  --model ae | svdd_frozen | svdd_ft   (same architectures and hyperparameters as 05 / 07)

Scores go to outputs/scores/ablation/<model>_<variant>_scores.csv (same columns as the main scores);
10_compare_models.py reports them in a separate table. Only validation metrics are printed here.
"""
import argparse
import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import (CLASSES, SCORE_COLUMNS, OCTDataset, get_device, get_split_frame, load_config, load_manifest,
                    load_splits, scoring_frame, set_seed, setup_logging, val_summary)
from nets import ConvAutoencoder, DeepSVDD, ResNetFeatures, frozen_embedding

ae_mod = importlib.import_module("05_train_autoencoder")
svdd_mod = importlib.import_module("07_train_deep_svdd")


# ------------------------------------------------------------ data
def naive_split(cfg, log):
    """Image-level split of main-experiment B-scans, created once."""
    path = Path(cfg["output_dir"]) / "splits_naive.csv"
    if path.exists():
        return pd.read_csv(path)
    m = load_manifest(cfg).merge(load_splits(cfg), on=["patient_id", "class_folder"])
    m = m[m.split.isin(["train", "val", "test"])]
    rng = np.random.default_rng(cfg["seed"])
    parts = []
    for cls in CLASSES:
        g = m[m.class_folder == cls]
        shares = g.split.value_counts(normalize=True).reindex(["train", "val", "test"], fill_value=0)
        order = rng.permutation(len(g))
        cuts = np.round(np.cumsum(shares.values) * len(g)).astype(int)
        naive = np.empty(len(g), dtype=object)
        for s, a, b in zip(shares.index, np.r_[0, cuts[:-1]], cuts):
            naive[order[a:b]] = s
        parts.append(pd.DataFrame(dict(image_path=g.image_path.values, split=naive)))
    sp = pd.concat(parts, ignore_index=True)
    sp.to_csv(path, index=False)
    log.info(f"naive image-level split -> {path}: {sp.split.value_counts().to_dict()}")
    return sp


def frames(cfg, variant, log):
    """(train, val, scoring) frames for the variant."""
    if variant == "expanded":
        extra = get_split_frame(cfg, "ablation_train", pool="normal_labelled")
        train = pd.concat([get_split_frame(cfg, "train"), extra], ignore_index=True)
        log.info(f"expanded pool: {len(train) - len(extra)} clean-normal + {len(extra)} normal-labelled B-scans from "
                 f"{extra.patient_id.nunique()} ablation_train DRUSEN/CNV patients")
        return train, get_split_frame(cfg, "val"), scoring_frame(cfg)
    sp = naive_split(cfg, log)
    m = load_manifest(cfg).merge(sp, on="image_path")
    clean = (m.class_folder == "NORMAL") & (m.bscan_label == "normal")
    train, val = m[(m.split == "train") & clean], m[(m.split == "val") & clean]
    score = m[m.split.isin(["val", "test"])][SCORE_COLUMNS].reset_index(drop=True)
    shared = set(train.patient_id) & set(m[m.split == "test"].patient_id)
    log.info(f"naive split: {len(train)} train / {len(val)} val clean-normal B-scans; {len(shared)} of "
             f"{m[m.split == 'test'].patient_id.nunique()} test patients also have B-scans in train")
    return train.reset_index(drop=True), val.reset_index(drop=True), score


# ---------------------------------------------------------- models
def train_ae(cfg, train, val, device, tag, log):
    hp = ae_mod.HP
    g = torch.Generator().manual_seed(cfg["seed"])
    tr_dl = DataLoader(OCTDataset(cfg, train, norm="unit", augment=True), batch_size=hp["batch_size"], shuffle=True,
                       generator=g, num_workers=0)
    va_dl = DataLoader(OCTDataset(cfg, val, norm="unit"), batch_size=hp["batch_size"], num_workers=0)
    model = ConvAutoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    best, wait, ckpt = float("inf"), 0, Path("models") / f"{tag}.pt"
    for epoch in range(1, hp["max_epochs"] + 1):
        t0 = time.time()
        tr = ae_mod.run_epoch(model, tr_dl, device, "mse", opt)
        va = ae_mod.run_epoch(model, va_dl, device, "mse")
        if va < best:
            best, wait = va, 0
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch, val_loss=va), ckpt)
        else:
            wait += 1
        log.info(f"epoch {epoch:3d}  train {tr:.6f}  val {va:.6f}  {time.time() - t0:5.1f}s" + ("  *" if wait == 0 else ""))
        if wait >= hp["patience"]:
            break
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    return model.eval()


@torch.no_grad()
def score_ae(model, cfg, df, device):
    dl = DataLoader(OCTDataset(cfg, df, norm="unit"), batch_size=64, num_workers=0)
    s = np.zeros(len(df))
    for x, i in dl:
        x = x.to(device)
        s[i.numpy()] = ((model(x) - x) ** 2).flatten(1).mean(1).cpu().numpy()
    return s


@torch.no_grad()
def score_frozen(cfg, train, df, device):
    dl = DataLoader(OCTDataset(cfg, train, norm="imagenet"), batch_size=64, num_workers=0)
    c, _ = svdd_mod.frozen_pass(dl, device)
    net = ResNetFeatures().to(device).eval()
    s = np.zeros(len(df))
    for x, i in DataLoader(OCTDataset(cfg, df, norm="imagenet"), batch_size=64, num_workers=0):
        l3, l4 = net(x.to(device))
        s[i.numpy()] = ((frozen_embedding(l3, l4) - c) ** 2).sum(1).cpu().numpy()
    return s


def train_score_svdd_ft(cfg, train, val, df, device, tag, log):
    hp = svdd_mod.HP
    plain = DataLoader(OCTDataset(cfg, train, norm="imagenet"), batch_size=64, num_workers=0)
    va_dl = DataLoader(OCTDataset(cfg, val, norm="imagenet"), batch_size=64, num_workers=0)
    model = DeepSVDD(hp["rep_dim"]).to(device)
    c, _ = svdd_mod.init_center(model, plain, device, hp["eps"])
    g = torch.Generator().manual_seed(cfg["seed"])
    tr_dl = DataLoader(OCTDataset(cfg, train, norm="imagenet", augment=True), batch_size=hp["batch_size"],
                       shuffle=True, generator=g, num_workers=0)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=hp["lr"],
                           weight_decay=hp["weight_decay"])
    best, wait, ckpt = svdd_mod.mean_distance(model, va_dl, c, device), 0, Path("models") / f"{tag}.pt"
    torch.save(dict(state_dict=model.state_dict(), epoch=0), ckpt)
    for epoch in range(1, hp["max_epochs"] + 1):
        t0 = time.time()
        model.train()
        for x, _ in tr_dl:
            loss = ((model(x.to(device)) - c) ** 2).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        va = svdd_mod.mean_distance(model, va_dl, c, device)
        if va < best:
            best, wait = va, 0
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch), ckpt)
        else:
            wait += 1
        log.info(f"epoch {epoch:3d}  val {va:.6f}  {time.time() - t0:5.1f}s" + ("  *" if wait == 0 else ""))
        if wait >= hp["patience"]:
            break
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    model.eval()
    s = np.zeros(len(df))
    with torch.no_grad():
        for x, i in DataLoader(OCTDataset(cfg, df, norm="imagenet"), batch_size=64, num_workers=0):
            s[i.numpy()] = ((model(x.to(device)) - c) ** 2).sum(1).cpu().numpy()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["expanded", "naive"], required=True)
    ap.add_argument("--model", choices=["ae", "svdd_frozen", "svdd_ft"], required=True)
    args = ap.parse_args()
    tag = f"{args.model}_{args.variant}"

    cfg = load_config()
    set_seed(cfg["seed"])
    log = setup_logging(f"13_ablation_{tag}", cfg)
    device = get_device(cfg["device"])
    out = Path(cfg["output_dir"]) / "scores" / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    train, val, df = frames(cfg, args.variant, log)
    t0 = time.time()
    if args.model == "ae":
        df["score"] = score_ae(train_ae(cfg, train, val, device, tag, log), cfg, df, device)
    elif args.model == "svdd_frozen":
        df["score"] = score_frozen(cfg, train, df, device)
    else:
        df["score"] = train_score_svdd_ft(cfg, train, val, df, device, tag, log)
    df.to_csv(out / f"{tag}_scores.csv", index=False)
    log.info(f"{tag}: scored {len(df)} B-scans in {(time.time() - t0) / 60:.1f} min -> {out / f'{tag}_scores.csv'}")
    val_summary(df, cfg, log, "score")


if __name__ == "__main__":
    main()
