"""v34: boost v20 with a second-stage Set Transformer on residuals.

Stage 1 = v20 OOF / test predictions (already at 0.197 OOF / 0.096 LB).
Stage 2 = Set Transformer trained with the SAME 5-fold scenario-grouped split
as v20, but predicting the residual r = y - v20_pred rather than y itself.

Key OOF-safety:
- Stage-1 OOF predictions used to compute residuals are already OOF (from
  v20's validation folds), so there is no leak.
- Stage 2 is trained 5-fold, each val fold's stage-2 OOF comes from a model
  that only saw residuals on the OTHER 4 folds.
- Final OOF = stage-1_oof + eta * stage-2_oof.

We sweep eta on the combined OOF and take the argmin, then apply the same
eta to the test predictions.  Full pipeline is validated before any
submission is produced.
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
    build_vocabs, load_properties, target_transform, target_inverse_transform,
)
from .augment_globals import attach_mcm_globals
from .train import train_fold, predict


def _clone(s: ScenarioSample, new_target: np.ndarray) -> ScenarioSample:
    d = {k: (v.copy() if isinstance(v, np.ndarray) else v)
         for k, v in s.__dict__.items()}
    d["targets"] = new_target.astype(np.float32)
    return ScenarioSample(**d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="artifacts_v20")
    ap.add_argument("--out_dir", default="artifacts_v34")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--seeds", type=int, default=3)
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

    # Load stage-1 (v20) OOF + test predictions
    scenarios = np.array([s.scenario_id for s in train_samples])
    test_ids = [s.scenario_id for s in test_samples]
    v1_oof_df = pd.read_csv(Path(args.base) / "oof_predictions.csv").set_index("scenario_id").loc[scenarios]
    v1_oof = v1_oof_df[["oof_viscosity", "oof_oxidation"]].values.astype(np.float32)
    v1_test_df = pd.read_csv(Path(args.base) / "predictions_raw_headers.csv").set_index("scenario_id").loc[test_ids]
    v1_test = v1_test_df[[
        "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
        "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"
    ]].values.astype(np.float32)

    std_v, std_o = y_tr[:, 0].std(), y_tr[:, 1].std()
    def norm(p, t):
        return np.mean(np.abs(p[:,0]-t[:,0]))/std_v/2 + np.mean(np.abs(p[:,1]-t[:,1]))/std_o/2
    base_norm = norm(v1_oof, y_tr)
    print(f"Stage-1 (v20) OOF norm = {base_norm:.4f}")

    # Compute residuals r = y - v1_oof (in raw space)
    r = y_tr - v1_oof
    print(f"Residual stats: mean=({r[:,0].mean():+.2f}, {r[:,1].mean():+.2f})  "
          f"std=({r[:,0].std():.2f}, {r[:,1].std():.2f})")

    G = np.stack([s.globals for s in train_samples], axis=0)
    global_dim_actual = int(G.shape[1])
    global_mu = G.mean(axis=0).astype(np.float32)
    global_sd = (G.std(axis=0) + 1e-6).astype(np.float32)

    # Build residual-target samples
    res_samples = [_clone(s, r[i]) for i, s in enumerate(train_samples)]

    # For stage 2 we need per-stage target stats based on transformed residuals
    y_t_res = target_transform(r)
    target_mu = y_t_res.mean(axis=0).astype(np.float32)
    target_sd = (y_t_res.std(axis=0) + 1e-6).astype(np.float32)

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    stage2_oof = np.zeros((len(train_samples), 2), dtype=np.float32)
    oof_count = np.zeros(len(train_samples), dtype=np.int32)
    stage2_test_sum = np.zeros((len(test_samples), 2), dtype=np.float32)
    n_models = 0
    device = torch.device(args.device)
    n_components = len(comp_vocab); n_types = len(type_vocab); n_props = len(TOP_PROPERTIES)

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(scenarios)):
        tr = [res_samples[i] for i in tr_idx]
        va = [res_samples[i] for i in va_idx]
        for seed in range(args.seeds):
            print(f"\n[stage2 fold {fold_idx+1}/{args.n_folds} seed {seed}]", flush=True)
            model, best_val, _ = train_fold(
                tr, va,
                n_components=n_components, n_types=n_types, n_props=n_props,
                condition_dim=CONDITION_DIM, global_dim_actual=global_dim_actual,
                global_mu=global_mu, global_sd=global_sd, device=device,
                target_mu=target_mu, target_sd=target_sd,
                epochs=args.epochs, batch_size=16,
                d_model=128, n_layers=3, dropout=0.20, comp_dropout=0.20,
                id_dropout=0.30, mass_aug=0.20, seed=seed * 100 + fold_idx,
            )
            print(f"  val_mae={best_val:.4f}")
            ids_va, preds_va = predict(model, va, target_mu, target_sd, device,
                                       global_mu=global_mu, global_sd=global_sd)
            for i, sid in enumerate(ids_va):
                j = int(np.where(scenarios == sid)[0][0])
                stage2_oof[j] += preds_va[i]; oof_count[j] += 1
            ids_te, preds_te = predict(model, test_samples, target_mu, target_sd, device,
                                       global_mu=global_mu, global_sd=global_sd)
            stage2_test_sum += preds_te
            n_models += 1
    stage2_oof = stage2_oof / np.clip(oof_count, 1, None)[:, None]
    stage2_test = stage2_test_sum / n_models

    print(f"\nStage-2 OOF residual MAE: v={np.mean(np.abs(stage2_oof[:,0])):.2f} "
          f"o={np.mean(np.abs(stage2_oof[:,1])):.2f}")
    # Sweep eta on OOF
    print("\n=== eta sweep on combined OOF ===")
    best_eta, best_norm = 0.0, base_norm
    for eta in np.linspace(-0.5, 1.5, 41):
        combined = v1_oof + eta * stage2_oof
        n = norm(combined, y_tr)
        if eta in (0.0, 0.5, 1.0):
            print(f"  eta={eta:+.2f}: norm={n:.4f}")
        if n < best_norm:
            best_norm = n; best_eta = float(eta)
    print(f"\nBest eta = {best_eta:+.3f} with OOF norm = {best_norm:.4f} (baseline {base_norm:.4f}, "
          f"delta {best_norm - base_norm:+.4f})")

    # Per-target eta (independent)
    print("\n=== per-target eta sweep ===")
    best_eta_v, best_eta_o = 0.0, 0.0
    best_sep = base_norm
    for ev in np.linspace(-0.5, 1.5, 41):
        for eo in np.linspace(-0.5, 1.5, 41):
            combined = v1_oof.copy()
            combined[:, 0] = combined[:, 0] + ev * stage2_oof[:, 0]
            combined[:, 1] = combined[:, 1] + eo * stage2_oof[:, 1]
            n = norm(combined, y_tr)
            if n < best_sep:
                best_sep = n; best_eta_v = float(ev); best_eta_o = float(eo)
    print(f"Best per-target eta: visc={best_eta_v:+.3f}, ox={best_eta_o:+.3f} -> OOF {best_sep:.4f} "
          f"(delta {best_sep - base_norm:+.4f})")

    # Apply the best per-target eta to the test predictions
    final_test = v1_test.copy()
    final_test[:, 0] = v1_test[:, 0] + best_eta_v * stage2_test[:, 0]
    final_test[:, 1] = v1_test[:, 1] + best_eta_o * stage2_test[:, 1]
    final_oof = v1_oof.copy()
    final_oof[:, 0] = v1_oof[:, 0] + best_eta_v * stage2_oof[:, 0]
    final_oof[:, 1] = v1_oof[:, 1] + best_eta_o * stage2_oof[:, 1]

    pd.DataFrame({
        "scenario_id": scenarios,
        "oof_viscosity": final_oof[:, 0], "oof_oxidation": final_oof[:, 1],
        "true_viscosity": y_tr[:, 0], "true_oxidation": y_tr[:, 1],
    }).to_csv(out_dir / "oof_predictions.csv", index=False)
    pd.DataFrame({
        "scenario_id": test_ids,
        "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %": final_test[:, 0],
        "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm": final_test[:, 1],
    }).to_csv(out_dir / "predictions_raw_headers.csv", index=False)
    with open(out_dir / "boost_summary.json", "w") as f:
        json.dump({"base_norm": float(base_norm),
                   "joint_eta": float(best_eta), "joint_norm": float(best_norm),
                   "eta_visc": best_eta_v, "eta_ox": best_eta_o,
                   "per_target_norm": float(best_sep),
                   "n_models": n_models}, f, indent=2)
    print(f"\nSaved v34 to {out_dir}/")


if __name__ == "__main__":
    main()
