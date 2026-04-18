"""v33: Set-Transformer boosting on residuals.

Train T1 with 5-fold scenario-grouped CV (OOF = y_hat_1).
Compute residual r_1 = y_true - y_hat_1 (in raw target space).
Train T2 on (x, r_1) with the SAME 5-fold CV (OOF = r_hat_1).
Final prediction per scenario: y_hat = y_hat_1 + eta * r_hat_1.
Optionally iterate to T3, T4, ...

Validation strictly via OOF: for each val fold, the stage-k model trains on
the OTHER folds' residuals-from-stage-(k-1) and predicts the val fold.  No
cross-fold leak.

Motivation: user specifically asked for transformer-boosting; each stage sees
the same features but a different target (the residual), which is a smaller-
variance signal that may be easier for a specialised model to capture.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from .data import (
    CONDITION_DIM, COL_COMP, TOP_PROPERTIES, GLOBAL_FEAT_DIM,
    ScenarioSample, build_component_property_table, build_scenario_samples,
    build_vocabs, load_properties, target_transform, target_inverse_transform,
)
from .augment_globals import attach_mcm_globals
from .train import train_fold, predict


def _clone(s: ScenarioSample, new_target: np.ndarray | None = None) -> ScenarioSample:
    d = {k: (v.copy() if isinstance(v, np.ndarray) else v)
         for k, v in s.__dict__.items()}
    if new_target is not None:
        d["targets"] = new_target.astype(np.float32)
    return ScenarioSample(**d)


def run_one_stage(train_samples, test_samples, y_targets,
                  n_components, n_types, n_props, global_dim_actual,
                  global_mu, global_sd, device,
                  n_folds=5, seeds=2, epochs=180,
                  d_model=160, n_layers=3, dropout=0.1, comp_dropout=0.15,
                  id_dropout=0.25, mass_aug=0.15):
    """Train one boosting stage.  y_targets is the per-sample target array
    (shape (N, 2) in raw target space) to fit this stage on.

    Returns (oof_pred, test_pred) both in raw target space.
    """
    # Override sample targets with y_targets
    fold_train_samples = [_clone(s, y_targets[i]) for i, s in enumerate(train_samples)]

    scenarios = np.array([s.scenario_id for s in train_samples])
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Per-stage target transform stats (asinh applied internally by train_fold).
    y_t = target_transform(y_targets)
    target_mu = y_t.mean(axis=0).astype(np.float32)
    target_sd = (y_t.std(axis=0) + 1e-6).astype(np.float32)

    oof = np.zeros((len(train_samples), 2), dtype=np.float32)
    oof_count = np.zeros(len(train_samples), dtype=np.int32)
    test_sum = np.zeros((len(test_samples), 2), dtype=np.float32)
    n_models = 0

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(scenarios)):
        tr = [fold_train_samples[i] for i in tr_idx]
        va = [fold_train_samples[i] for i in va_idx]
        for seed in range(seeds):
            print(f"  stage fold {fold_idx+1}/{n_folds} seed {seed}", flush=True)
            model, best_val, _ = train_fold(
                tr, va,
                n_components=n_components, n_types=n_types, n_props=n_props,
                condition_dim=CONDITION_DIM, global_dim_actual=global_dim_actual,
                global_mu=global_mu, global_sd=global_sd, device=device,
                target_mu=target_mu, target_sd=target_sd,
                epochs=epochs, batch_size=16,
                d_model=d_model, n_layers=n_layers,
                dropout=dropout, comp_dropout=comp_dropout,
                id_dropout=id_dropout, mass_aug=mass_aug,
                seed=seed * 100 + fold_idx,
            )
            # predict() returns raw target space (after inverse transform)
            ids_va, preds_va = predict(model, va, target_mu, target_sd, device,
                                       global_mu=global_mu, global_sd=global_sd)
            for i, sid in enumerate(ids_va):
                j = int(np.where(scenarios == sid)[0][0])
                oof[j] += preds_va[i]; oof_count[j] += 1
            ids_te, preds_te = predict(model, test_samples, target_mu, target_sd, device,
                                       global_mu=global_mu, global_sd=global_sd)
            test_sum += preds_te
            n_models += 1
    oof = oof / np.clip(oof_count, 1, None)[:, None]
    return oof, test_sum / n_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="artifacts_v33")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n_stages", type=int, default=3)
    ap.add_argument("--eta", type=float, default=0.5, help="Shrinkage per stage.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)

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

    G = np.stack([s.globals for s in train_samples], axis=0)
    global_dim_actual = int(G.shape[1])
    global_mu = G.mean(axis=0).astype(np.float32)
    global_sd = (G.std(axis=0) + 1e-6).astype(np.float32)

    device = torch.device(args.device)
    n_components = len(comp_vocab); n_types = len(type_vocab); n_props = len(TOP_PROPERTIES)

    std_v, std_o = y_tr[:, 0].std(), y_tr[:, 1].std()
    def norm(p, t):
        return np.mean(np.abs(p[:,0]-t[:,0]))/std_v/2 + np.mean(np.abs(p[:,1]-t[:,1]))/std_o/2

    scenarios = np.array([s.scenario_id for s in train_samples])
    oof_running = np.zeros_like(y_tr)
    test_running = np.zeros((len(test_samples), 2), dtype=np.float32)
    per_stage = []

    # Stage 1: fit target y_tr directly
    current_target = y_tr.copy()
    stage_configs = [
        dict(d_model=160, n_layers=3, dropout=0.10, comp_dropout=0.15,
             id_dropout=0.25, mass_aug=0.15),
        dict(d_model=128, n_layers=2, dropout=0.08, comp_dropout=0.05,
             id_dropout=0.10, mass_aug=0.05),
        dict(d_model=96,  n_layers=2, dropout=0.05, comp_dropout=0.02,
             id_dropout=0.05, mass_aug=0.02),
        dict(d_model=64,  n_layers=2, dropout=0.05, comp_dropout=0.0,
             id_dropout=0.0, mass_aug=0.0),
    ]

    for stage in range(args.n_stages):
        print(f"\n===== Stage {stage+1}/{args.n_stages} =====", flush=True)
        cfg = stage_configs[min(stage, len(stage_configs)-1)]
        oof_s, test_s = run_one_stage(
            train_samples, test_samples, current_target,
            n_components=n_components, n_types=n_types, n_props=n_props,
            global_dim_actual=global_dim_actual,
            global_mu=global_mu, global_sd=global_sd, device=device,
            n_folds=args.n_folds, seeds=args.seeds, epochs=args.epochs,
            **cfg,
        )
        eta_this = 1.0 if stage == 0 else args.eta
        oof_running = oof_running + eta_this * oof_s
        test_running = test_running + eta_this * test_s
        current_target = y_tr - oof_running  # residual for next stage

        stage_norm = norm(oof_running, y_tr)
        per_stage.append({"stage": stage+1, "eta": float(eta_this),
                          "oof_norm": float(stage_norm),
                          "cfg": {k: float(v) if isinstance(v, (int, float)) else v for k, v in cfg.items()}})
        print(f"Stage {stage+1} cumulative OOF norm: {stage_norm:.4f}", flush=True)
        # Save intermediate after every stage so we have checkpoints.
        pd.DataFrame({
            "scenario_id": scenarios,
            "oof_viscosity": oof_running[:, 0], "oof_oxidation": oof_running[:, 1],
            "true_viscosity": y_tr[:, 0], "true_oxidation": y_tr[:, 1],
        }).to_csv(out_dir / f"oof_stage{stage+1}.csv", index=False)
        pd.DataFrame({
            "scenario_id": [s.scenario_id for s in test_samples],
            "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %": test_running[:, 0],
            "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm": test_running[:, 1],
        }).to_csv(out_dir / f"predictions_stage{stage+1}.csv", index=False)
        with open(out_dir / "progress.json", "w") as f:
            json.dump(per_stage, f, indent=2)

    # Final outputs are the deepest stage
    print("\n=== Per-stage cumulative OOF ===")
    for s in per_stage:
        print(f"  stage {s['stage']} (eta={s['eta']:.2f}): OOF norm = {s['oof_norm']:.4f}")
    # Write the BEST stage as the official output (often stage 1 or 2).
    best = min(per_stage, key=lambda s: s["oof_norm"])
    print(f"\nBEST stage = {best['stage']} with OOF norm {best['oof_norm']:.4f}")
    import shutil
    shutil.copy(out_dir / f"oof_stage{best['stage']}.csv", out_dir / "oof_predictions.csv")
    shutil.copy(out_dir / f"predictions_stage{best['stage']}.csv", out_dir / "predictions_raw_headers.csv")
    print(f"Saved BEST stage predictions to {out_dir}/predictions_raw_headers.csv")


if __name__ == "__main__":
    main()
