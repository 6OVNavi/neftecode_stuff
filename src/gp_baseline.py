"""v27: Gaussian Process + Kernel Ridge baselines on rich tabular features.

Radically different model family from Set Transformer: closed-form / kernel-based
rather than attention-based NN. On n=167 small-data regimes, GPs frequently beat
deep models and the Open Polymer Challenge report calls out ridge/KRR stacks in
top-5 solutions.

Produces 5-fold OOF + test predictions, saves to artifacts_v27_{gp,krr}/ and
packages both into submission zips.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern

from .data import (
    CONDITION_DIM, COL_COMP, TOP_PROPERTIES, GLOBAL_FEAT_DIM,
    build_component_property_table, build_scenario_samples, build_vocabs,
    load_properties, target_transform, target_inverse_transform,
)
from .tabular_rich import build_tabular_rich
from .augment_globals import attach_mcm_globals


def run(X_tr, y_tr, X_te, kind, seed=0):
    """Fit a single (X, y) regressor per target and predict on X_te."""
    pred_te = np.zeros((X_te.shape[0], 2), dtype=np.float32)
    if kind == "gp":
        kernel = (ConstantKernel(1.0, (1e-2, 10.0)) *
                  Matern(length_scale=1.0, length_scale_bounds=(1e-1, 1e3), nu=2.5)
                  + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-3, 10.0)))
        for j in range(2):
            m = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True,
                                         n_restarts_optimizer=3, random_state=seed)
            m.fit(X_tr, y_tr[:, j])
            pred_te[:, j] = m.predict(X_te)
    elif kind == "krr":
        for j in range(2):
            m = KernelRidge(kernel="rbf", alpha=0.3, gamma=1.0 / X_tr.shape[1])
            m.fit(X_tr, y_tr[:, j])
            pred_te[:, j] = m.predict(X_te)
    elif kind == "krr_poly":
        for j in range(2):
            m = KernelRidge(kernel="polynomial", alpha=1.0, degree=3, coef0=1.0)
            m.fit(X_tr, y_tr[:, j])
            pred_te[:, j] = m.predict(X_te)
    else:
        raise ValueError(kind)
    return pred_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="gp", choices=["gp", "krr", "krr_poly"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)

    mix_train = pd.read_csv("daimler_mixtures_train.csv")
    mix_test = pd.read_csv("daimler_mixtures_test.csv")
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

    X_tr_raw, names, y_raw, ids_tr = build_tabular_rich(train_samples, mix_train, mix_test, y_tr)
    X_te_raw, _, _, ids_te = build_tabular_rich(test_samples, mix_train, mix_test, y_tr)
    print(f"Feature dim: {X_tr_raw.shape[1]} ({len(names)} names)")

    # Replace NaN/inf, standardise
    X_tr_raw = np.nan_to_num(X_tr_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_raw = np.nan_to_num(X_te_raw, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw).astype(np.float32)
    X_te = scaler.transform(X_te_raw).astype(np.float32)

    # target transform for heavy tails
    y_t = target_transform(y_tr)
    target_mu = y_t.mean(axis=0); target_sd = y_t.std(axis=0) + 1e-6
    y_tr_z = (y_t - target_mu) / target_sd

    # 5-fold CV
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    oof = np.zeros((len(train_samples), 2), dtype=np.float32)
    test_pred_sum = np.zeros((len(test_samples), 2), dtype=np.float32)
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
        pred_va = run(X_tr[tr_idx], y_tr_z[tr_idx], X_tr[va_idx], args.kind, seed=fold_idx)
        # invert normalisation
        pred_va = pred_va * target_sd + target_mu
        pred_va_raw = target_inverse_transform(pred_va)
        oof[va_idx] = pred_va_raw
        # test
        pred_te = run(X_tr[tr_idx], y_tr_z[tr_idx], X_te, args.kind, seed=fold_idx)
        pred_te = pred_te * target_sd + target_mu
        pred_te_raw = target_inverse_transform(pred_te)
        test_pred_sum += pred_te_raw
        print(f"  fold {fold_idx}: val MAE v={np.mean(np.abs(pred_va_raw[:,0]-y_tr[va_idx,0])):.2f} "
              f"o={np.mean(np.abs(pred_va_raw[:,1]-y_tr[va_idx,1])):.2f}")

    test_pred = test_pred_sum / args.n_folds
    mv = np.mean(np.abs(oof[:,0]-y_tr[:,0]))
    mo = np.mean(np.abs(oof[:,1]-y_tr[:,1]))
    std_v = y_tr[:,0].std(); std_o = y_tr[:,1].std()
    norm = mv/std_v/2 + mo/std_o/2
    print(f"\n=== {args.kind} OOF: visc {mv:.3f}, ox {mo:.3f}, norm {norm:.4f} ===")

    pd.DataFrame({
        "scenario_id": ids_tr,
        "oof_viscosity": oof[:,0], "oof_oxidation": oof[:,1],
        "true_viscosity": y_tr[:,0], "true_oxidation": y_tr[:,1],
    }).to_csv(out_dir / "oof_predictions.csv", index=False)
    pd.DataFrame({
        "scenario_id": ids_te,
        "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %": test_pred[:,0],
        "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm": test_pred[:,1],
    }).to_csv(out_dir / "predictions_raw_headers.csv", index=False)
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
