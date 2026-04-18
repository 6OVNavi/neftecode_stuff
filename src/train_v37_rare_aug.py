"""v37: rare-condition targeted Dirichlet augmentation.

Condition imbalance diagnosis: T=160 bio=0 dominates with 115/167 train rows;
T=150 bio=7 has only 27 combined.  MAE_v is ~10 in the dominant regime and
~100 in the rare ones.  Prior global balancing via sample weights (v31) hurt
overall performance because upweighted loss caused instability in the common
regime.

This variant INSTEAD augments only the rare-condition rows by creating k
Dirichlet-perturbed mass clones per rare scenario (keeping comp_ids, props,
conditions and targets), inflating rare regimes to roughly the size of the
dominant one while leaving the dominant regime untouched.

The augmentation is done BEFORE CV and applied only to the training fold at
each split, so the validation fold never contains augmented copies.

Validated on 5-fold scenario-grouped OOF against v20 (0.1976) and v16
(0.1904).
"""
from __future__ import annotations
import argparse
import copy
import json
from collections import Counter
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


COL_TEMP = "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C"
COL_TIME = "Время испытания | - Daimler Oxidation Test (DOT), ч"
COL_BIO = "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс"


def cond_of(sample_id, mix_train):
    r = mix_train[mix_train.scenario_id == sample_id].iloc[0]
    return (float(r[COL_TEMP]), float(r[COL_TIME]), float(r[COL_BIO]))


def dirichlet_perturb_sample(s: ScenarioSample, alpha: float, rng: np.random.Generator,
                             suffix: str) -> ScenarioSample:
    """Clone sample with masses resampled from Dirichlet(alpha * original_mass)."""
    n = len(s.mass)
    m_conc = alpha * np.maximum(s.mass, 1e-3)  # concentration ~ original masses
    new_mass = rng.dirichlet(m_conc).astype(np.float32)
    new = ScenarioSample(
        scenario_id=f"{s.scenario_id}_aug{suffix}",
        comp_ids=s.comp_ids.copy(), type_ids=s.type_ids.copy(),
        props=s.props.copy(), miss_mask=s.miss_mask.copy(),
        mass=new_mass,
        conditions=s.conditions.copy(),
        is_new=s.is_new.copy(),
        globals=s.globals.copy(),  # will be same; conditions are what matter at scenario level
        targets=s.targets.copy() if s.targets is not None else None,
        weight=s.weight * 0.6,  # downweight synthetic copies
    )
    return new


def augment_rare(samples, mix_train, target_count=80, rng=None):
    """Create copies of rare-condition samples so each (T, t, bio) bucket has
    at least `target_count` rows.  Returns augmented list; original samples
    are untouched."""
    if rng is None:
        rng = np.random.default_rng(2026)
    by_cond = {}
    for s in samples:
        c = cond_of(s.scenario_id, mix_train)
        by_cond.setdefault(c, []).append(s)

    out = list(samples)
    added = 0
    for c, group in by_cond.items():
        shortage = target_count - len(group)
        if shortage <= 0:
            continue
        per_sample = int(np.ceil(shortage / len(group)))
        print(f"  regime {c}: n={len(group)}, creating {per_sample}x Dirichlet clones -> +{per_sample * len(group)}")
        for s in group:
            for k in range(per_sample):
                alpha = rng.uniform(5.0, 40.0)  # higher alpha = smaller perturbation
                out.append(dirichlet_perturb_sample(s, alpha, rng, f"{c}_{k}"))
                added += 1
    print(f"Augmentation added {added} synthetic rows (total {len(out)})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="artifacts_v37")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--target_count", type=int, default=60,
                    help="Minimum samples per condition bucket after augmentation")
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

    # Standardise globals on ORIGINAL train
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

    rng = np.random.default_rng(2026)
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(scenarios)):
        tr_samples = [train_samples[i] for i in tr_idx]
        va_samples = [train_samples[i] for i in va_idx]
        print(f"\n[v37 fold {fold_idx+1}/{args.n_folds}] augmenting rare conds (target={args.target_count})")
        tr_aug = augment_rare(tr_samples, mix_train, target_count=args.target_count, rng=rng)
        for seed in range(args.seeds):
            print(f"  seed {seed}", flush=True)
            model, best_val, _ = train_fold(
                tr_aug, va_samples,
                n_components=n_components, n_types=n_types, n_props=n_props,
                condition_dim=CONDITION_DIM, global_dim_actual=global_dim_actual,
                global_mu=global_mu, global_sd=global_sd, device=device,
                target_mu=target_mu, target_sd=target_sd,
                epochs=args.epochs, batch_size=16,
                d_model=160, n_layers=3, dropout=0.10, comp_dropout=0.15,
                id_dropout=0.25, mass_aug=0.10, seed=seed * 100 + fold_idx,
            )
            print(f"    val_mae={best_val:.4f}")
            summary.append(dict(fold=fold_idx, seed=seed, val_mae=best_val,
                                n_train_aug=len(tr_aug)))
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
    print(f"\n=== v37 OOF: visc {mv:.3f}, ox {mo:.3f}, norm {mv/std_v/2 + mo/std_o/2:.4f} ===")

    # Per-condition breakdown
    cond_list = np.array([cond_of(s.scenario_id, mix_train) for s in train_samples])
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
    print(f"Saved v37 to {out_dir}/")


if __name__ == "__main__":
    main()
