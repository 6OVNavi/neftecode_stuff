"""v35: boost v20 with FT-Transformer (tabular) on residuals.

Stage 1 = v20 OOF / test predictions.
Stage 2 = FT-Transformer trained on 142 rich tabular features, target =
residual r = y - v20_pred.  Different architecture AND different features
than v20, so stage 2 may capture signal stage 1 missed.

Validated on 5-fold OOF, eta swept per-target, result kept only if it
honestly beats v20 (0.1976).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .data import (
    COL_COMP, build_component_property_table, build_scenario_samples,
    build_vocabs, load_properties, target_transform, target_inverse_transform,
)
from .tabular_rich import build_tabular_rich
from .augment_globals import attach_mcm_globals
from .ft_transformer import FTTransformer, train_fold_ft


def fit_ft_stage(X_tr, r_tr, X_va, X_te, epochs=200, seeds=3):
    """Return (oof_va_raw, te_mean_raw) for residual target r_tr in RAW space."""
    # Transform residual for numerical stability
    r_t = target_transform(r_tr)
    mu = r_t.mean(axis=0).astype(np.float32)
    sd = (r_t.std(axis=0) + 1e-6).astype(np.float32)
    y_norm = (r_t - mu) / sd

    X_va_t = torch.from_numpy(X_va).float()
    X_te_t = torch.from_numpy(X_te).float()

    oof_sum = np.zeros((len(X_va), 2), dtype=np.float32)
    te_sum = np.zeros((len(X_te), 2), dtype=np.float32)
    for seed in range(seeds):
        model, best = train_fold_ft(X_tr, y_norm, X_va, np.zeros((len(X_va), 2), dtype=np.float32),
                                    epochs=epochs, seed=seed,
                                    pseudo_X=None, pseudo_y=None, pseudo_w=0.5)
        model.eval()
        with torch.no_grad():
            p_va = model(X_va_t).numpy()
            p_te = model(X_te_t).numpy()
        # invert normalisation + asinh
        p_va_raw = target_inverse_transform(p_va * sd + mu)
        p_te_raw = target_inverse_transform(p_te * sd + mu)
        oof_sum += p_va_raw
        te_sum += p_te_raw
    return oof_sum / seeds, te_sum / seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="artifacts_v20")
    ap.add_argument("--out_dir", default="artifacts_v35")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)

    mix_train = pd.read_csv("daimler_mixtures_train.csv")
    mix_test = pd.read_csv("daimler_mixtures_test.csv")
    mix_test.columns = mix_test.columns.str.strip("\ufeff")
    pr = load_properties("daimler_component_properties.csv")
    wide_batch, wide_comp, mu_props, sd_props = build_component_property_table(pr)
    comp_vocab, type_vocab = build_vocabs(mix_train, mix_test)
    train_samples = build_scenario_samples(mix_train, wide_batch, wide_comp, mu_props, sd_props,
                                           comp_vocab, type_vocab,
                                           train_comp_set=set(mix_train[COL_COMP].unique()), is_train=True)
    test_samples = build_scenario_samples(mix_test, wide_batch, wide_comp, mu_props, sd_props,
                                          comp_vocab, type_vocab,
                                          train_comp_set=set(mix_train[COL_COMP].unique()), is_train=False)
    y_tr = np.stack([s.targets for s in train_samples])
    attach_mcm_globals(train_samples, test_samples, mix_train, mix_test, y_tr,
                       n_latent_svd=12, n_latent_nmf=8)

    X_tr_raw, _, _, ids_tr = build_tabular_rich(train_samples, mix_train, mix_test, y_tr)
    X_te_raw, _, _, ids_te = build_tabular_rich(test_samples, mix_train, mix_test, y_tr)
    X_tr_raw = np.nan_to_num(X_tr_raw); X_te_raw = np.nan_to_num(X_te_raw)
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw).astype(np.float32)
    X_te = scaler.transform(X_te_raw).astype(np.float32)

    scenarios = np.array([s.scenario_id for s in train_samples])
    v1 = pd.read_csv(Path(args.base) / "oof_predictions.csv").set_index("scenario_id").loc[scenarios]
    v1_oof = v1[["oof_viscosity", "oof_oxidation"]].values.astype(np.float32)
    v1_test_df = pd.read_csv(Path(args.base) / "predictions_raw_headers.csv").set_index("scenario_id").loc[ids_te]
    v1_test = v1_test_df[[
        "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
        "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm",
    ]].values.astype(np.float32)
    std_v, std_o = y_tr[:,0].std(), y_tr[:,1].std()
    def norm(p, t): return np.mean(np.abs(p[:,0]-t[:,0]))/std_v/2 + np.mean(np.abs(p[:,1]-t[:,1]))/std_o/2
    base_norm = norm(v1_oof, y_tr)
    print(f"v20 OOF: {base_norm:.4f}")

    residuals = y_tr - v1_oof

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    stage2_oof = np.zeros_like(y_tr)
    stage2_test_sum = np.zeros((len(test_samples), 2), dtype=np.float32)
    n_folds_done = 0
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
        print(f"\n[v35 fold {fold_idx+1}/{args.n_folds}]", flush=True)
        oof_va, test_avg = fit_ft_stage(
            X_tr[tr_idx], residuals[tr_idx],
            X_tr[va_idx], X_te,
            epochs=args.epochs, seeds=args.seeds,
        )
        stage2_oof[va_idx] = oof_va
        stage2_test_sum += test_avg
        n_folds_done += 1
    stage2_test = stage2_test_sum / n_folds_done

    print(f"\nStage-2 FT OOF residual MAE: v={np.mean(np.abs(stage2_oof[:,0])):.2f} "
          f"o={np.mean(np.abs(stage2_oof[:,1])):.2f}  "
          f"(true residual MAE v={np.mean(np.abs(residuals[:,0])):.2f} "
          f"o={np.mean(np.abs(residuals[:,1])):.2f})")

    # eta sweep
    print("\n=== eta sweep ===")
    best_eta, best_norm = 0.0, base_norm
    for eta in np.linspace(-1.0, 1.5, 51):
        combined = v1_oof + eta * stage2_oof
        n = norm(combined, y_tr)
        if abs(eta) < 0.01 or eta in (0.5, 1.0):
            print(f"  eta={eta:+.3f}: {n:.4f}")
        if n < best_norm:
            best_norm = n; best_eta = float(eta)
    # Per-target
    best_vx, best_ox, best_sep = 0.0, 0.0, base_norm
    for ev in np.linspace(-1.0, 1.5, 26):
        for eo in np.linspace(-1.0, 1.5, 26):
            combined = v1_oof.copy()
            combined[:,0] += ev * stage2_oof[:,0]
            combined[:,1] += eo * stage2_oof[:,1]
            n = norm(combined, y_tr)
            if n < best_sep:
                best_sep = n; best_vx = float(ev); best_ox = float(eo)
    print(f"\nBest joint eta={best_eta:+.3f}: OOF {best_norm:.4f} (delta {best_norm-base_norm:+.4f})")
    print(f"Best per-target eta_v={best_vx:+.3f} eta_o={best_ox:+.3f}: OOF {best_sep:.4f} "
          f"(delta {best_sep-base_norm:+.4f})")

    final_oof = v1_oof.copy(); final_test = v1_test.copy()
    final_oof[:,0] += best_vx * stage2_oof[:,0]; final_oof[:,1] += best_ox * stage2_oof[:,1]
    final_test[:,0] += best_vx * stage2_test[:,0]; final_test[:,1] += best_ox * stage2_test[:,1]

    pd.DataFrame({
        "scenario_id": scenarios,
        "oof_viscosity": final_oof[:,0], "oof_oxidation": final_oof[:,1],
        "true_viscosity": y_tr[:,0], "true_oxidation": y_tr[:,1],
    }).to_csv(out_dir / "oof_predictions.csv", index=False)
    pd.DataFrame({
        "scenario_id": ids_te,
        "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %": final_test[:,0],
        "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm": final_test[:,1],
    }).to_csv(out_dir / "predictions_raw_headers.csv", index=False)
    with open(out_dir / "boost_summary.json", "w") as f:
        json.dump({"base_norm": float(base_norm),
                   "best_joint_eta": float(best_eta), "joint_norm": float(best_norm),
                   "best_eta_v": best_vx, "best_eta_o": best_ox,
                   "per_target_norm": float(best_sep),
                   "beats_baseline": bool(best_sep < base_norm - 0.001)}, f, indent=2)
    print(f"Saved v35 to {out_dir}/")


if __name__ == "__main__":
    main()
