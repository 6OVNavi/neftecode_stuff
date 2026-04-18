"""v38: LubriSet with Arrhenius-parameterized oxidation head.

The viscosity head is unchanged.  The oxidation head predicts (log A, Ea/R,
AO_depletion) per scenario, then computes
    EOT = exp(log A - Ea/R · 1/T_K) · t · (1 - AO)
with T_K = T_C + 273.15, t in hours.  Both (T_C, t) come from the condition
vector.

This adds a physics inductive bias ONLY on the oxidation target where
Arrhenius kinetics are physically appropriate; viscosity change is left to
the flexible MLP head.  Physical bounds on (log A, Ea/R, AO) are enforced
through tanh / sigmoid gating so the network cannot produce non-sensical
parameters.

Only constant used is the conversion T_C → T_K (+273.15).  The ideal-gas
constant R is absorbed into the learned Ea/R parameter.

Validated on 5-fold scenario-grouped OOF against v20 (0.1976) and v16
(0.1904).
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
    build_vocabs, load_properties, target_transform,
)
from .augment_globals import attach_mcm_globals
from .train import train_fold, predict
from .model_arrhenius import LubriSetArrhenius


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="artifacts_v38")
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

    G = np.stack([s.globals for s in train_samples], axis=0)
    global_dim_actual = int(G.shape[1])
    global_mu = G.mean(axis=0).astype(np.float32)
    global_sd = (G.std(axis=0) + 1e-6).astype(np.float32)

    y_t = target_transform(y_tr)
    target_mu = y_t.mean(axis=0).astype(np.float32)
    target_sd = (y_t.std(axis=0) + 1e-6).astype(np.float32)

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
            print(f"\n[v38 fold {fold_idx+1}/{args.n_folds} seed {seed}]", flush=True)
            model, best_val, _ = train_fold(
                tr_samples, va_samples,
                n_components=n_components, n_types=n_types, n_props=n_props,
                condition_dim=CONDITION_DIM, global_dim_actual=global_dim_actual,
                global_mu=global_mu, global_sd=global_sd, device=device,
                target_mu=target_mu, target_sd=target_sd,
                epochs=args.epochs, batch_size=16,
                d_model=160, n_layers=3, dropout=0.10, comp_dropout=0.15,
                id_dropout=0.25, mass_aug=0.15, seed=seed * 100 + fold_idx,
                model_cls=LubriSetArrhenius,
            )
            print(f"  val_mae={best_val:.4f}")
            summary.append(dict(fold=fold_idx, seed=seed, val_mae=best_val))
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
    print(f"\n=== v38 OOF: visc {mv:.3f}, ox {mo:.3f}, norm {mv/std_v/2 + mo/std_o/2:.4f} ===")

    # Per-condition breakdown
    COL_TEMP = "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C"
    COL_TIME = "Время испытания | - Daimler Oxidation Test (DOT), ч"
    COL_BIO = "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс"
    cond_dict = {sid: (float(r[COL_TEMP]), float(r[COL_TIME]), float(r[COL_BIO]))
                 for sid, r in mix_train.groupby("scenario_id").first().iterrows()}
    cond_list = np.array([cond_dict[s] for s in scenarios])
    print("\n=== Per-condition OOF MAE ===")
    for c in sorted(set(map(tuple, cond_list))):
        m = np.all(cond_list == np.array(c), axis=1)
        if m.sum() < 2: continue
        print(f"  T={c[0]:.0f} t={c[1]:.0f} bio={c[2]:.0f}: n={m.sum()} "
              f"MAE_v={np.mean(np.abs(err[m, 0])):.1f} MAE_o={np.mean(np.abs(err[m, 1])):.1f}")

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
    print(f"Saved v38 to {out_dir}/")


if __name__ == "__main__":
    main()
