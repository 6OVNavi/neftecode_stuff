"""v32: aggressive per-target transforms.

Diagnosis showed extreme tails dominate OOF error:
  - T=160 bio=0  ΔKV100 range ~ [-35, +30]      MAE_v ~ 10
  - T=150 bio=7  ΔKV100 range ~ [+64, +1763]    MAE_v ~ 103

Current asinh(y/10) under-weights extreme samples.  v32 tries a more aggressive
monotonic transform:
  - Viscosity: sign(y) * log1p(|y| / 5)   (compresses 1000 → 5.3)
  - Oxidation: log1p(y / 10)              (y is always >= 0)

Everything else held constant vs v20 baseline.  We validate on 5-fold OOF and
compare to v20 0.1970 / v16 0.1898.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from .data import (
    CONDITION_DIM, COL_COMP, TOP_PROPERTIES, GLOBAL_FEAT_DIM,
    ScenarioSample, build_component_property_table, build_scenario_samples,
    build_vocabs, load_properties,
)
import src.data as dm
from .augment_globals import attach_mcm_globals
from .train import train_fold, predict


def log_transform(y):
    y = np.asarray(y, dtype=np.float32).copy()
    y[:, 0] = np.sign(y[:, 0]) * np.log1p(np.abs(y[:, 0]) / 5.0)
    y[:, 1] = np.log1p(np.maximum(y[:, 1], 0.0) / 10.0)
    return y


def log_inverse(y_t):
    y = np.asarray(y_t, dtype=np.float32).copy()
    y[:, 0] = np.sign(y[:, 0]) * (np.expm1(np.abs(y[:, 0])) * 5.0)
    y[:, 1] = np.expm1(np.maximum(y[:, 1], 0.0)) * 10.0
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="artifacts_v32")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)

    # Monkey-patch target transforms in the data module so downstream code uses
    # ours transparently.  predict() calls target_inverse_transform(asinh ...)
    # but we need it to call our inverse.
    dm.target_transform = log_transform
    dm.target_inverse_transform = log_inverse
    # The train module has already imported target_transform at import time;
    # force rebind there too.
    import src.train as tm
    tm.target_transform = log_transform
    tm.target_inverse_transform = log_inverse

    mix_train = pd.read_csv("daimler_mixtures_train.csv")
    mix_test = pd.read_csv("daimler_mixtures_test.csv")
    mix_test.columns = mix_test.columns.str.strip("\ufeff")
    pr = load_properties("daimler_component_properties.csv")
    wide_batch, wide_comp, mu, sd = build_component_property_table(pr)
    comp_vocab, type_vocab = build_vocabs(mix_train, mix_test)
    train_samples = build_scenario_samples(
        mix_train, wide_batch, wide_comp, mu, sd, comp_vocab, type_vocab,
        train_comp_set=set(mix_train[COL_COMP].unique()), is_train=True)
    test_samples = build_scenario_samples(
        mix_test, wide_batch, wide_comp, mu, sd, comp_vocab, type_vocab,
        train_comp_set=set(mix_train[COL_COMP].unique()), is_train=False)
    y_tr = np.stack([s.targets for s in train_samples])
    attach_mcm_globals(train_samples, test_samples, mix_train, mix_test, y_tr,
                       n_latent_svd=12, n_latent_nmf=8)

    y_t = log_transform(y_tr)
    target_mu = y_t.mean(axis=0).astype(np.float32)
    target_sd = (y_t.std(axis=0) + 1e-6).astype(np.float32)
    print(f"Log-transformed target stats: mu={target_mu}, sd={target_sd}")
    G = np.stack([s.globals for s in train_samples], axis=0)
    global_dim_actual = int(G.shape[1])
    global_mu = G.mean(axis=0).astype(np.float32)
    global_sd = (G.std(axis=0) + 1e-6).astype(np.float32)

    scenarios = np.array([s.scenario_id for s in train_samples])
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(train_samples), 2), dtype=np.float32)
    oof_count = np.zeros(len(train_samples), dtype=np.int32)
    test_preds_sum = np.zeros((len(test_samples), 2), dtype=np.float32)
    n_models = 0
    device = torch.device(args.device)
    n_components = len(comp_vocab); n_types = len(type_vocab); n_props = len(TOP_PROPERTIES)
    summary = []

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(scenarios)):
        tr_samples = [train_samples[i] for i in tr_idx]
        va_samples = [train_samples[i] for i in va_idx]
        for seed in range(args.seeds):
            print(f"\n[v32 fold {fold_idx+1}/{args.n_folds} seed {seed}]")
            model, best_val, _ = train_fold(
                tr_samples, va_samples,
                n_components=n_components, n_types=n_types, n_props=n_props,
                condition_dim=CONDITION_DIM, global_dim_actual=global_dim_actual,
                global_mu=global_mu, global_sd=global_sd, device=device,
                target_mu=target_mu, target_sd=target_sd,
                epochs=args.epochs, batch_size=16,
                d_model=160, n_layers=3, dropout=0.10, comp_dropout=0.15,
                id_dropout=0.25, mass_aug=0.15, seed=seed * 100 + fold_idx,
            )
            summary.append(dict(fold=fold_idx, seed=seed, val_mae=best_val))
            print(f"  val_mae={best_val:.4f}")
            ids_va, preds_va = predict(model, va_samples, target_mu, target_sd, device,
                                       global_mu=global_mu, global_sd=global_sd)
            for i, sid in enumerate(ids_va):
                j = int(np.where(scenarios == sid)[0][0])
                oof_preds[j] += preds_va[i]; oof_count[j] += 1
            ids_te, preds_te = predict(model, test_samples, target_mu, target_sd, device,
                                       global_mu=global_mu, global_sd=global_sd)
            test_preds_sum += preds_te
            n_models += 1

    oof_avg = oof_preds / np.clip(oof_count, 1, None)[:, None]
    err = oof_avg - y_tr
    mv = float(np.mean(np.abs(err[:, 0]))); mo = float(np.mean(np.abs(err[:, 1])))
    std_v = y_tr[:, 0].std(); std_o = y_tr[:, 1].std()
    print(f"\n=== v32 OOF: visc {mv:.3f}, ox {mo:.3f}, norm {mv/std_v/2 + mo/std_o/2:.4f} ===")

    pd.DataFrame({
        "scenario_id": scenarios,
        "oof_viscosity": oof_avg[:, 0], "oof_oxidation": oof_avg[:, 1],
        "true_viscosity": y_tr[:, 0], "true_oxidation": y_tr[:, 1],
    }).to_csv(out_dir / "oof_predictions.csv", index=False)
    test_preds = test_preds_sum / n_models
    pd.DataFrame({
        "scenario_id": [s.scenario_id for s in test_samples],
        "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %": test_preds[:, 0],
        "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm": test_preds[:, 1],
    }).to_csv(out_dir / "predictions_raw_headers.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved v32 to {out_dir}/")


if __name__ == "__main__":
    main()
