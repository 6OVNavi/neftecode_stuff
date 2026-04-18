"""Synthetic-mixture pretraining for LubriSet.

Builds tens of thousands of fake lubricant mixtures whose targets follow simple
physics laws (Walther viscosity blending + Arrhenius oxidation kinetics with
antioxidant depletion), pretrains a Set Transformer on them, then fine-tunes
on the 167 real scenarios.

Motivation (from 57-paper synthesis):
- Simulation as Supervision (2507.08977): diverse mechanistic simulators as a
  pretraining signal beat zero-shot, especially for small-n real datasets.
- TabForestPFN (2405.13396): when foundation-model weights are unavailable,
  train your own ICL transformer from scratch on a synthetic prior that
  matches the downstream task's structure.
- With mixtures-of-components we get a uniquely faithful prior: physical
  blending rules are cheap to simulate.

Usage:
    python -m src.synth_pretrain --n_synth 50000 --pretrain_epochs 30 \
        --finetune_epochs 200 --out_dir artifacts_v25
"""
from __future__ import annotations
import argparse
import copy
import json
import math
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
from .train import train_fold, predict
from .augment_globals import attach_mcm_globals


# -----------------------------------------------------------------------------
# Synthetic mixture generator
# -----------------------------------------------------------------------------

def _pool_from_samples(samples: list[ScenarioSample]):
    """Collect unique (comp_id, type_id, props, miss_mask) tuples seen in train."""
    seen = {}
    for s in samples:
        for i in range(len(s.comp_ids)):
            cid = int(s.comp_ids[i])
            if cid in seen:
                continue
            seen[cid] = {
                "comp_id": cid,
                "type_id": int(s.type_ids[i]),
                "props": s.props[i].copy(),
                "miss": s.miss_mask[i].copy(),
                "is_new": int(s.is_new[i]),
            }
    return list(seen.values())


def _mix_globals(type_ids, props, miss, mass, global_mu, global_sd):
    """Cheap surrogate for compute_global_features on synthetic data.

    Uses mass-weighted sums of standardised props (real props are already
    standardised on disk) and a count vector per type.  Dimensions are padded
    to match GLOBAL_FEAT_DIM so the pretrained weights stay compatible.
    """
    # mass-weighted standardised props
    obs = (1.0 - miss) * props  # zero out missing
    w = mass[:, None]
    pooled = (w * obs).sum(axis=0)  # (P,)
    coverage = (w * (1.0 - miss)).sum(axis=0)  # (P,)

    out = np.zeros(GLOBAL_FEAT_DIM, dtype=np.float32)
    k = min(len(pooled), GLOBAL_FEAT_DIM // 2)
    out[:k] = pooled[:k]
    out[k:2 * k] = coverage[:k]
    return out


def synth_mixture(pool, rng, target_mu, target_sd,
                  global_mu=None, global_sd=None):
    """Draw one synthetic scenario with physics-inspired targets.

    Targets are produced in the *transformed* target space (asinh) so they can
    be compared directly with real training targets by train_fold().
    """
    n = int(rng.integers(6, 21))
    idx = rng.choice(len(pool), size=n, replace=False)
    mems = [pool[i] for i in idx]

    comp_ids = np.array([m["comp_id"] for m in mems], dtype=np.int64)
    type_ids = np.array([m["type_id"] for m in mems], dtype=np.int64)
    props = np.stack([m["props"] for m in mems], axis=0).astype(np.float32)
    miss = np.stack([m["miss"] for m in mems], axis=0).astype(np.float32)
    is_new = np.array([m["is_new"] for m in mems], dtype=np.int64)

    # Dirichlet masses with per-mixture concentration
    alpha = rng.uniform(0.3, 3.0)
    mass = rng.dirichlet(np.full(n, alpha)).astype(np.float32)

    # Conditions: T, time, biofuel, catalyst category -- ranges picked to
    # cover the real empirical distribution generously.
    T_C = rng.uniform(130.0, 170.0)
    t_h = rng.uniform(120.0, 400.0)
    biofuel = rng.choice([0.0, 5.0, 10.0], p=[0.6, 0.3, 0.1])
    cat = rng.integers(0, 3)
    conditions = np.array([T_C, t_h, biofuel, float(cat)], dtype=np.float32)
    if CONDITION_DIM > 4:
        conditions = np.concatenate([conditions,
                                     np.zeros(CONDITION_DIM - 4, dtype=np.float32)])

    # -------- physics-inspired pretext targets --------------------------
    # We operate on standardised props directly; physics becomes dimensionless
    # which is fine for a pretext signal.

    # First few TOP_PROPERTIES are: KV100, KV40, P, Ca, Zn, S, VI, CCS30 ...
    # treat prop[0] as "viscosity proxy" and prop[6] as "VI proxy".
    visc_proxy = props[:, 0]            # (n,)
    phos_proxy = props[:, 2]            # mass-action oxidation catalyst
    calc_proxy = props[:, 3]            # detergent basic reserve
    zinc_proxy = props[:, 4]            # antiwear/antioxidant synergy
    sulfur = props[:, 5]
    # antioxidant efficacy: heuristic from type + Zn + Ca + nitrogen (prop 31 if present)
    # Type id 2 = antioxidant (per COMPONENT_TYPES order)
    ao_mask = (type_ids == 2).astype(np.float32)

    # Walther-style log-log blend of viscosity
    log_nu = visc_proxy
    blended_log_nu = float((mass * log_nu).sum())

    # Arrhenius-ish rate (all dimensionless units, but shape matches real curve)
    Ea = rng.uniform(0.3, 1.5)          # activation barrier, dimensionless
    A = rng.uniform(0.5, 3.0)
    # map T to a [0,1] scaled temperature then Arrhenius with random Ea
    T_norm = (T_C - 130.0) / 40.0
    k_ox = A * math.exp(-Ea * (1.0 - T_norm))

    # AO reserve = mass-weighted AO-typed components * (1 + standardised N/Zn)
    ao_reserve = float((mass * ao_mask * (1.0 + 0.5 * zinc_proxy)).sum())

    # sulfur/phosphorus burden accelerates oxidation (or retards it per additive pkg)
    s_p = float((mass * (sulfur + 0.3 * phos_proxy + 0.3 * calc_proxy)).sum())

    # EOT ~ cumulative oxidation over time, saturating when AO exhausted
    raw_eot = k_ox * (t_h / 200.0) * (1.0 - math.tanh(2.5 * ao_reserve)) \
              + 0.4 * s_p * (t_h / 200.0)
    # ΔKV100 driven by EOT (viscosity thickens as oxidation proceeds) + base drift
    raw_dkv = 30.0 * raw_eot + 3.0 * blended_log_nu \
              + rng.normal(0.0, 2.0)

    # EOT to compare against real (A/cm); scale and add noise
    raw_eot = 80.0 + 30.0 * raw_eot + rng.normal(0.0, 3.0)
    raw_dkv = raw_dkv + rng.normal(0.0, 2.0)

    raw = np.array([raw_dkv, raw_eot], dtype=np.float32)
    targets = target_transform(raw[None])[0].astype(np.float32)  # asinh transformed

    # globals (physics aggregate) surrogate
    g = _mix_globals(type_ids, props, miss, mass, global_mu, global_sd)

    return ScenarioSample(
        scenario_id=f"synth_{rng.integers(0, 2**31):x}",
        comp_ids=comp_ids, type_ids=type_ids,
        props=props, miss_mask=miss, mass=mass,
        conditions=conditions, is_new=is_new, globals=g,
        targets=targets, weight=1.0,
    )


def generate_synth(n, pool, rng, target_mu, target_sd,
                   global_mu=None, global_sd=None):
    out = []
    for _ in range(n):
        out.append(synth_mixture(pool, rng, target_mu, target_sd,
                                 global_mu, global_sd))
    return out


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_synth", type=int, default=50000)
    ap.add_argument("--pretrain_epochs", type=int, default=25)
    ap.add_argument("--finetune_epochs", type=int, default=160)
    ap.add_argument("--out_dir", default="artifacts_v25")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=202604)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)

    # Load real data --------------------------------------------------------
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
    K = attach_mcm_globals(train_samples, test_samples,
                           mix_train, mix_test, y_tr,
                           n_latent_svd=12, n_latent_nmf=8)
    print(f"MCM attached: +{K} dims")

    y_t = target_transform(y_tr)
    target_mu = y_t.mean(axis=0).astype(np.float32)
    target_sd = (y_t.std(axis=0) + 1e-6).astype(np.float32)
    G = np.stack([s.globals for s in train_samples], axis=0)
    global_dim_actual = int(G.shape[1])
    global_mu = G.mean(axis=0).astype(np.float32)
    global_sd = (G.std(axis=0) + 1e-6).astype(np.float32)

    # Generate synthetic pool ----------------------------------------------
    pool = _pool_from_samples(train_samples)
    print(f"Component pool: {len(pool)} unique components")

    rng = np.random.default_rng(args.seed)
    synth_samples = generate_synth(args.n_synth, pool, rng,
                                   target_mu, target_sd, global_mu, global_sd)
    # pad synthetic globals to match real dimensionality
    pad = global_dim_actual - synth_samples[0].globals.shape[0]
    if pad > 0:
        for s in synth_samples:
            s.globals = np.concatenate([s.globals,
                                        np.zeros(pad, dtype=np.float32)])
    elif pad < 0:
        for s in synth_samples:
            s.globals = s.globals[:global_dim_actual].astype(np.float32)
    print(f"Generated {len(synth_samples)} synthetic mixtures")

    device = torch.device(args.device)
    n_components = len(comp_vocab); n_types = len(type_vocab); n_props = len(TOP_PROPERTIES)

    # Stage 1: pretrain on synthetic with no real val split ----------------
    # We borrow train_fold but feed synth as train and a tiny holdout as val.
    synth_tr = synth_samples[: int(0.98 * len(synth_samples))]
    synth_va = synth_samples[int(0.98 * len(synth_samples)):]
    print("\n>>> Stage 1: pretrain on synthetic")
    pretrain_model, pretrain_val, _ = train_fold(
        synth_tr, synth_va,
        n_components=n_components, n_types=n_types, n_props=n_props,
        condition_dim=CONDITION_DIM, global_dim_actual=global_dim_actual,
        global_mu=global_mu, global_sd=global_sd, device=device,
        target_mu=target_mu, target_sd=target_sd,
        epochs=args.pretrain_epochs, batch_size=64,
        d_model=160, n_layers=3, dropout=0.1, comp_dropout=0.1,
        id_dropout=0.10, mass_aug=0.0, seed=args.seed,
    )
    print(f"  pretrain synth_val_mae={pretrain_val:.4f}")
    torch.save(pretrain_model.state_dict(), out_dir / "pretrain.pt")

    # Stage 2: fine-tune on 167 real with 5-fold CV -----------------------
    scenarios = np.array([s.scenario_id for s in train_samples])
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(train_samples), 2), dtype=np.float32)
    oof_count = np.zeros(len(train_samples), dtype=np.int32)
    test_preds_sum = np.zeros((len(test_samples), 2), dtype=np.float32)
    n_models = 0
    val_summary = []

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(scenarios)):
        tr_samples = [train_samples[i] for i in tr_idx]
        va_samples = [train_samples[i] for i in va_idx]
        for seed in [0, 1, 2]:
            print(f"\n[Fold {fold_idx+1}/{args.n_folds} seed {seed}] fine-tune")
            model, best_val, _ = train_fold(
                tr_samples, va_samples,
                n_components=n_components, n_types=n_types, n_props=n_props,
                condition_dim=CONDITION_DIM, global_dim_actual=global_dim_actual,
                global_mu=global_mu, global_sd=global_sd, device=device,
                target_mu=target_mu, target_sd=target_sd,
                epochs=args.finetune_epochs, batch_size=16,
                d_model=160, n_layers=3, dropout=0.1, comp_dropout=0.15,
                id_dropout=0.25, mass_aug=0.2,
                seed=seed * 1000 + fold_idx,
                init_state_dict=pretrain_model.state_dict(),
                lr=5e-4,  # lower LR on fine-tune
            )
            val_summary.append({'fold': fold_idx, 'seed': seed, 'val_mae': best_val})
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
    print(f"\n=== v25 OOF: visc {mv:.3f}, ox {mo:.3f}, norm {mv/std_v/2 + mo/std_o/2:.4f} ===")

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
    with open(out_dir / "val_summary.json", "w") as f:
        json.dump(val_summary, f, indent=2)
    print(f"\nSaved v25 to {out_dir}/")


if __name__ == "__main__":
    main()
