"""Train a diverse ensemble of Set Transformers with varied hyperparameters.

For each fold, train multiple models with DIFFERENT:
- d_model (96, 128, 160)
- n_layers (2, 3, 4)
- dropout (0.05, 0.10, 0.15)
- comp_dropout (0.10, 0.15, 0.20)
- mass_aug (0.0, 0.15, 0.3)
- seeds

This gives much more ensemble diversity than replicating same config.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from .data import (
    CONDITION_DIM, COL_COMP, TOP_PROPERTIES, GLOBAL_FEAT_DIM,
    ScenarioSample, build_component_property_table,
    build_scenario_samples, build_vocabs, load_properties,
    target_transform, target_inverse_transform,
)
from .train import train_fold, predict, SetDataset
from .augment_globals import attach_mcm_globals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="artifacts_v20")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--pseudo_csv", default=None)
    ap.add_argument("--pseudo_weight", type=float, default=0.5)
    ap.add_argument("--mcm", action="store_true")
    ap.add_argument("--mcm_svd", type=int, default=12)
    ap.add_argument("--mcm_nmf", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)

    mix_train = pd.read_csv("daimler_mixtures_train.csv")
    mix_test = pd.read_csv("daimler_mixtures_test.csv")
    pr = load_properties("daimler_component_properties.csv")
    wide_batch, wide_comp, mu, sd = build_component_property_table(pr)
    comp_vocab, type_vocab = build_vocabs(mix_train, mix_test)
    train_samples_all = build_scenario_samples(
        mix_train, wide_batch, wide_comp, mu, sd, comp_vocab, type_vocab,
        train_comp_set=set(mix_train[COL_COMP].unique()), is_train=True)
    test_samples = build_scenario_samples(
        mix_test, wide_batch, wide_comp, mu, sd, comp_vocab, type_vocab,
        train_comp_set=set(mix_train[COL_COMP].unique()), is_train=False)

    if args.mcm:
        y_tr = np.stack([s.targets for s in train_samples_all])
        K = attach_mcm_globals(train_samples_all, test_samples,
                                mix_train, mix_test, y_tr,
                                n_latent_svd=args.mcm_svd, n_latent_nmf=args.mcm_nmf)
        print(f"Attached MCM features: +{K} dims per scenario global vector")

    pseudo_samples = []
    if args.pseudo_csv:
        pc = pd.read_csv(args.pseudo_csv)
        pc = pc.rename(columns={pc.columns[1]: "target_viscosity",
                                pc.columns[2]: "target_oxidation"})
        by_id = {r["scenario_id"]: (r["target_viscosity"], r["target_oxidation"])
                 for _, r in pc.iterrows()}
        for ts in test_samples:
            if ts.scenario_id not in by_id: continue
            y1, y2 = by_id[ts.scenario_id]
            ps = ScenarioSample(
                scenario_id=ts.scenario_id, comp_ids=ts.comp_ids, type_ids=ts.type_ids,
                props=ts.props, miss_mask=ts.miss_mask, mass=ts.mass,
                conditions=ts.conditions, is_new=ts.is_new, globals=ts.globals,
                targets=np.array([y1, y2], dtype=np.float32),
                weight=float(args.pseudo_weight),
            )
            pseudo_samples.append(ps)
        print(f"Loaded {len(pseudo_samples)} pseudo rows, weight {args.pseudo_weight}")

    y_raw = np.stack([s.targets for s in train_samples_all], axis=0)
    y_t = target_transform(y_raw)
    target_mu = y_t.mean(axis=0).astype(np.float32)
    target_sd = (y_t.std(axis=0) + 1e-6).astype(np.float32)
    G = np.stack([s.globals for s in train_samples_all], axis=0)
    global_dim_actual = int(G.shape[1])
    global_mu = G.mean(axis=0).astype(np.float32)
    global_sd = (G.std(axis=0) + 1e-6).astype(np.float32)

    device = torch.device(args.device)
    n_components = len(comp_vocab); n_types = len(type_vocab); n_props = len(TOP_PROPERTIES)
    print(f"Components={n_components}, Types={n_types}, Props={n_props}, Global={global_dim_actual}")

    # Diverse configurations — 6 per fold
    configs = [
        dict(d_model=128, n_layers=3, dropout=0.10, comp_dropout=0.15, mass_aug=0.15, seed=0),
        dict(d_model=160, n_layers=3, dropout=0.12, comp_dropout=0.10, mass_aug=0.00, seed=1),
        dict(d_model=96,  n_layers=4, dropout=0.08, comp_dropout=0.20, mass_aug=0.30, seed=2),
        dict(d_model=128, n_layers=2, dropout=0.15, comp_dropout=0.15, mass_aug=0.20, seed=3),
        dict(d_model=128, n_layers=3, dropout=0.05, comp_dropout=0.10, mass_aug=0.10, seed=4),
        dict(d_model=160, n_layers=4, dropout=0.10, comp_dropout=0.15, mass_aug=0.25, seed=5),
    ]

    scenarios = np.array([s.scenario_id for s in train_samples_all])
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(train_samples_all), 2), dtype=np.float32)
    oof_count = np.zeros(len(train_samples_all), dtype=np.int32)
    test_preds_sum = np.zeros((len(test_samples), 2), dtype=np.float32)
    n_models = 0
    val_summary = []
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(scenarios)):
        tr_samples = [train_samples_all[i] for i in tr_idx]
        va_samples = [train_samples_all[i] for i in va_idx]
        if pseudo_samples:
            tr_samples = tr_samples + pseudo_samples
        for cfg_idx, cfg in enumerate(configs):
            print(f"\n[Fold {fold_idx+1}/{args.n_folds} cfg {cfg_idx}] {cfg}", flush=True)
            model, best_val, _ = train_fold(
                tr_samples, va_samples,
                n_components=n_components, n_types=n_types, n_props=n_props,
                condition_dim=CONDITION_DIM,
                global_dim_actual=global_dim_actual,
                global_mu=global_mu, global_sd=global_sd, device=device,
                target_mu=target_mu, target_sd=target_sd,
                epochs=args.epochs, batch_size=16,
                d_model=cfg['d_model'], n_layers=cfg['n_layers'],
                dropout=cfg['dropout'], comp_dropout=cfg['comp_dropout'],
                mass_aug=cfg['mass_aug'], seed=cfg['seed'],
            )
            val_summary.append({'fold': fold_idx, 'cfg': cfg_idx, 'val_mae': best_val, **cfg})
            print(f"  val_mae={best_val:.4f}")
            ids_va, preds_va = predict(model, va_samples, target_mu, target_sd, device,
                                       global_mu=global_mu, global_sd=global_sd)
            for i, sid in enumerate(ids_va):
                j = int(np.where(scenarios == sid)[0][0])
                oof_preds[j] += preds_va[i]
                oof_count[j] += 1
            ids_te, preds_te = predict(model, test_samples, target_mu, target_sd, device,
                                        global_mu=global_mu, global_sd=global_sd)
            test_preds_sum += preds_te
            n_models += 1
            torch.save({
                "state_dict": model.state_dict(),
                "target_mu": target_mu, "target_sd": target_sd,
                "global_mu": global_mu, "global_sd": global_sd,
                "comp_vocab": comp_vocab, "type_vocab": type_vocab, "mu": mu, "sd": sd,
                "config": {"d_model": cfg['d_model'], "n_layers": cfg['n_layers'],
                           "n_components": n_components, "n_types": n_types, "n_props": n_props,
                           "condition_dim": CONDITION_DIM, "global_dim": global_dim_actual},
            }, out_dir / f"model_fold{fold_idx}_cfg{cfg_idx}.pt")

    oof_avg = oof_preds / np.clip(oof_count, 1, None)[:, None]
    err = oof_avg - y_raw
    mv = float(np.mean(np.abs(err[:,0]))); mo = float(np.mean(np.abs(err[:,1])))
    print(f"\n=== OOF raw: visc {mv:.3f}, ox {mo:.3f}, norm {mv/y_raw[:,0].std()/2 + mo/y_raw[:,1].std()/2:.4f} ===")

    oof_df = pd.DataFrame({
        "scenario_id": scenarios,
        "oof_viscosity": oof_avg[:,0], "oof_oxidation": oof_avg[:,1],
        "true_viscosity": y_raw[:,0], "true_oxidation": y_raw[:,1],
    })
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    test_preds = test_preds_sum / n_models
    ids_test = [s.scenario_id for s in test_samples]
    sub = pd.DataFrame({
        "scenario_id": ids_test,
        "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %": test_preds[:,0],
        "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm": test_preds[:,1],
    })
    sub.to_csv(out_dir / "predictions_raw_headers.csv", index=False)
    print(f"Saved {n_models} models, OOF, predictions to {out_dir}")
    with open(out_dir / "val_summary.json", "w") as f:
        json.dump(val_summary, f, indent=2)

if __name__ == "__main__":
    main()
