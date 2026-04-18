"""v22: even more diverse configs, orthogonal to v20.

v20 used 6 configs. v22 picks 8 DIFFERENT configs to extend coverage:
- bigger d_model (192)
- very shallow (n_layers=1)
- very deep (n_layers=5)
- high id_dropout for better generalization on new components
- pseudo labels from different sources (v20, v16, none)
- MCM on/off mix
- different target transform scales (asinh scale 5 vs 20 for visc)

Goal: make 40 models whose errors are as uncorrelated as possible with v20's.
Final = mean of (v20 30 models + v22 40 models).
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
    ScenarioSample, build_component_property_table,
    build_scenario_samples, build_vocabs, load_properties,
    target_transform, target_inverse_transform,
)
from .train import train_fold, predict, SetDataset
from .augment_globals import attach_mcm_globals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="artifacts_v22")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=220)
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

    # Attach MCM (needed for any config using MCM)
    y_tr = np.stack([s.targets for s in train_samples_all])
    K = attach_mcm_globals(train_samples_all, test_samples,
                            mix_train, mix_test, y_tr,
                            n_latent_svd=12, n_latent_nmf=8)
    print(f"Attached MCM features: +{K} dims per scenario global vector")

    # Load all pseudo sources
    def load_pseudo(path, weight=0.5):
        pc = pd.read_csv(path)
        pc = pc.rename(columns={pc.columns[1]: "target_viscosity",
                                pc.columns[2]: "target_oxidation"})
        by_id = {r["scenario_id"]: (r["target_viscosity"], r["target_oxidation"])
                 for _, r in pc.iterrows()}
        out = []
        for ts in test_samples:
            if ts.scenario_id not in by_id: continue
            y1, y2 = by_id[ts.scenario_id]
            ps = ScenarioSample(
                scenario_id=ts.scenario_id, comp_ids=ts.comp_ids, type_ids=ts.type_ids,
                props=ts.props, miss_mask=ts.miss_mask, mass=ts.mass,
                conditions=ts.conditions, is_new=ts.is_new, globals=ts.globals,
                targets=np.array([y1, y2], dtype=np.float32),
                weight=float(weight),
            )
            out.append(ps)
        return out

    # Pseudo sources: use v20 (best LB 0.0964), v16 (best OOF), and none.
    pseudo_v20 = load_pseudo('artifacts_v20/predictions_raw_headers.csv', 0.5)
    pseudo_v16 = load_pseudo('artifacts_v16/predictions_raw_headers.csv', 0.5)
    print(f"Loaded pseudo_v20: {len(pseudo_v20)}, pseudo_v16: {len(pseudo_v16)}")

    y_t = target_transform(y_tr)
    target_mu = y_t.mean(axis=0).astype(np.float32)
    target_sd = (y_t.std(axis=0) + 1e-6).astype(np.float32)
    G = np.stack([s.globals for s in train_samples_all], axis=0)
    global_dim_actual = int(G.shape[1])
    global_mu = G.mean(axis=0).astype(np.float32)
    global_sd = (G.std(axis=0) + 1e-6).astype(np.float32)

    device = torch.device(args.device)
    n_components = len(comp_vocab); n_types = len(type_vocab); n_props = len(TOP_PROPERTIES)

    # 8 diverse configs — ORTHOGONAL to v20's configs (different combinations)
    configs = [
        # (d_model, n_layers, dropout, comp_dropout, mass_aug, seed, pseudo_label, id_dropout)
        dict(d_model=192, n_layers=3, dropout=0.10, comp_dropout=0.15, mass_aug=0.15, seed=100, pseudo='v20', id_dropout=0.25),
        dict(d_model=128, n_layers=5, dropout=0.15, comp_dropout=0.20, mass_aug=0.10, seed=101, pseudo='v16', id_dropout=0.30),
        dict(d_model=192, n_layers=2, dropout=0.08, comp_dropout=0.25, mass_aug=0.30, seed=102, pseudo='v20', id_dropout=0.20),
        dict(d_model=160, n_layers=3, dropout=0.15, comp_dropout=0.05, mass_aug=0.05, seed=103, pseudo=None, id_dropout=0.35),
        dict(d_model=128, n_layers=4, dropout=0.12, comp_dropout=0.25, mass_aug=0.35, seed=104, pseudo='v16', id_dropout=0.20),
        dict(d_model=96,  n_layers=3, dropout=0.05, comp_dropout=0.15, mass_aug=0.20, seed=105, pseudo='v20', id_dropout=0.40),
        dict(d_model=160, n_layers=5, dropout=0.10, comp_dropout=0.10, mass_aug=0.00, seed=106, pseudo='v16', id_dropout=0.25),
        dict(d_model=128, n_layers=3, dropout=0.20, comp_dropout=0.20, mass_aug=0.20, seed=107, pseudo='v20', id_dropout=0.15),
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
        for cfg_idx, cfg in enumerate(configs):
            # Pick pseudo
            if cfg['pseudo'] == 'v20':
                tr_samples_cfg = tr_samples + pseudo_v20
            elif cfg['pseudo'] == 'v16':
                tr_samples_cfg = tr_samples + pseudo_v16
            else:
                tr_samples_cfg = tr_samples
            print(f"\n[Fold {fold_idx+1}/{args.n_folds} cfg {cfg_idx}] {cfg}", flush=True)
            model, best_val, _ = train_fold(
                tr_samples_cfg, va_samples,
                n_components=n_components, n_types=n_types, n_props=n_props,
                condition_dim=CONDITION_DIM,
                global_dim_actual=global_dim_actual,
                global_mu=global_mu, global_sd=global_sd, device=device,
                target_mu=target_mu, target_sd=target_sd,
                epochs=args.epochs, batch_size=16,
                d_model=cfg['d_model'], n_layers=cfg['n_layers'],
                dropout=cfg['dropout'], comp_dropout=cfg['comp_dropout'],
                id_dropout=cfg['id_dropout'], mass_aug=cfg['mass_aug'], seed=cfg['seed'],
            )
            val_summary.append({'fold': fold_idx, 'cfg': cfg_idx, 'val_mae': best_val, **{k:v for k,v in cfg.items() if k != 'pseudo'}, 'pseudo': cfg['pseudo']})
            print(f"  val_mae={best_val:.4f}", flush=True)
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
                "config": {"d_model": cfg['d_model'], "n_layers": cfg['n_layers'],
                           "n_components": n_components, "n_types": n_types, "n_props": n_props,
                           "condition_dim": CONDITION_DIM, "global_dim": global_dim_actual},
            }, out_dir / f"model_fold{fold_idx}_cfg{cfg_idx}.pt")

    oof_avg = oof_preds / np.clip(oof_count, 1, None)[:, None]
    err = oof_avg - y_tr
    mv = float(np.mean(np.abs(err[:,0]))); mo = float(np.mean(np.abs(err[:,1])))
    print(f"\n=== OOF raw: visc {mv:.3f}, ox {mo:.3f}, norm {mv/y_tr[:,0].std()/2 + mo/y_tr[:,1].std()/2:.4f} ===")

    oof_df = pd.DataFrame({
        "scenario_id": scenarios,
        "oof_viscosity": oof_avg[:,0], "oof_oxidation": oof_avg[:,1],
        "true_viscosity": y_tr[:,0], "true_oxidation": y_tr[:,1],
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
