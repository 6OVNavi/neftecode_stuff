"""v28: retrieval-augmented Set Transformer.

For each scenario (train and test) we find up to k training scenarios whose
component-mass composition is most similar, and append their
  [present, match_quality, anchor_conds, Δconds, anchor_targets_asinh]
as extra global features.  The anchor is drawn only from the current training
fold (no OOF leak during CV) and from all of train at inference.

Motivation (validated):
- 47 of 167 train scenarios have a same-composition sibling in train.
- 22 of 40 test scenarios have a same-composition row in train.
- v20 OOF on those 47 has v=68 o=8; the sibling-ground-truth noise floor is
  v=0.2 o=0.4.  That is the size of the signal the model is wasting.

Approach is additive: existing architecture is untouched, only `globals` gains
anchor dims.  Scenarios without a sibling get a zero slot + zero flag so the
model learns to ignore the anchor when it is absent.
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
    build_vocabs, load_properties, target_transform,
)
from .augment_globals import attach_mcm_globals
from .train import train_fold, predict


# Per-anchor slot dims:
# [present, jaccard, T, t, bio, cat_oh(3), dT, dt, dbio, dcat, y1_asinh, y2_asinh]
SLOT = 1 + 1 + 6 + 4 + 2


def _sig(df, mass_round=1):
    out = {}
    for sid, grp in df.groupby("scenario_id"):
        out[sid] = frozenset((row["Компонент"], round(row["Массовая доля, %"], mass_round))
                             for _, row in grp.iterrows())
    return out


def _conds(df):
    COL_TEMP = "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C"
    COL_TIME = "Время испытания | - Daimler Oxidation Test (DOT), ч"
    COL_BIO = "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс"
    COL_CAT = "Дозировка катализатора, категория"
    out = {}
    for sid, grp in df.groupby("scenario_id"):
        r = grp.iloc[0]
        out[sid] = (float(r[COL_TEMP]), float(r[COL_TIME]),
                    float(r[COL_BIO]), float(r[COL_CAT]))
    return out


def _anchor_slot(my_sid, my_sig, my_cond, candidate_pool, k=1):
    """candidate_pool: list of (sid, sig, cond, target_raw_float32_2). Returns
    k * SLOT dim array."""
    scored = []
    for cid, c_sig, c_cond, c_target in candidate_pool:
        if cid == my_sid:
            continue
        overlap = len(my_sig & c_sig)
        union = len(my_sig | c_sig)
        jac = overlap / max(union, 1)
        if jac < 0.5:
            continue
        dT = abs(my_cond[0] - c_cond[0])
        dt = abs(my_cond[1] - c_cond[1])
        dbio = abs(my_cond[2] - c_cond[2])
        dcat = 0 if my_cond[3] == c_cond[3] else 1
        cond_dist = dT / 10.0 + dt / 48.0 + dbio / 5.0 + dcat * 0.3
        score = jac * 100 - cond_dist
        scored.append((score, jac, dT, dt, dbio, dcat, cid, c_cond, c_target))
    scored.sort(key=lambda x: -x[0])

    slots = []
    for i in range(k):
        if i < len(scored):
            _, jac, dT, dt, dbio, dcat, _, c_cond, c_target = scored[i]
            cat_oh = np.zeros(3, dtype=np.float32)
            cat_oh[int(c_cond[3]) % 3] = 1.0
            y_t = target_transform(np.array(c_target, dtype=np.float32)[None])[0]
            slot = np.concatenate([
                [1.0, jac],
                [c_cond[0], c_cond[1], c_cond[2]], cat_oh,
                [float(my_cond[0] - c_cond[0]),
                 float(my_cond[1] - c_cond[1]),
                 float(my_cond[2] - c_cond[2]),
                 0.0 if my_cond[3] == c_cond[3] else 1.0],
                y_t.astype(np.float32),
            ]).astype(np.float32)
        else:
            slot = np.zeros(SLOT, dtype=np.float32)
        slots.append(slot)
    return np.concatenate(slots, axis=0).astype(np.float32)


def attach_anchors(train_samples, test_samples, mix_train, mix_test, y_tr,
                   candidate_indices, k=1, mass_round=1):
    """Append k anchor slots to each sample's globals in place.

    - candidate_indices: iterable of positional indices in train_samples that
      may be used as anchors (OOF-safe: only current fold's TR indices).
    - At test, anchors may come from ANY train sample (we ignore
      candidate_indices for test).
    """
    sig_tr = _sig(mix_train, mass_round)
    sig_te = _sig(mix_test, mass_round)
    cond_tr = _conds(mix_train)
    cond_te = _conds(mix_test)

    cand_set = set(int(i) for i in candidate_indices)
    tr_pool = [(train_samples[i].scenario_id, sig_tr[train_samples[i].scenario_id],
                cond_tr[train_samples[i].scenario_id], y_tr[i])
               for i in cand_set]
    full_pool = [(train_samples[i].scenario_id, sig_tr[train_samples[i].scenario_id],
                  cond_tr[train_samples[i].scenario_id], y_tr[i])
                 for i in range(len(train_samples))]

    for s in train_samples:
        slot = _anchor_slot(s.scenario_id, sig_tr[s.scenario_id],
                            cond_tr[s.scenario_id], tr_pool, k=k)
        s.globals = np.concatenate([s.globals, slot]).astype(np.float32)
    for s in test_samples:
        slot = _anchor_slot(s.scenario_id, sig_te[s.scenario_id],
                            cond_te[s.scenario_id], full_pool, k=k)
        s.globals = np.concatenate([s.globals, slot]).astype(np.float32)
    return k * SLOT


def _clone_sample(s: ScenarioSample) -> ScenarioSample:
    return ScenarioSample(**{k: (v.copy() if isinstance(v, np.ndarray) else v)
                             for k, v in s.__dict__.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="artifacts_v28")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)

    mix_train = pd.read_csv("daimler_mixtures_train.csv")
    mix_test = pd.read_csv("daimler_mixtures_test.csv")
    mix_test.columns = mix_test.columns.str.strip("\ufeff")
    pr = load_properties("daimler_component_properties.csv")
    wide_batch, wide_comp, mu, sd = build_component_property_table(pr)
    comp_vocab, type_vocab = build_vocabs(mix_train, mix_test)

    base_train = build_scenario_samples(
        mix_train, wide_batch, wide_comp, mu, sd, comp_vocab, type_vocab,
        train_comp_set=set(mix_train[COL_COMP].unique()), is_train=True)
    base_test = build_scenario_samples(
        mix_test, wide_batch, wide_comp, mu, sd, comp_vocab, type_vocab,
        train_comp_set=set(mix_train[COL_COMP].unique()), is_train=False)

    y_tr = np.stack([s.targets for s in base_train])
    mcm_k = attach_mcm_globals(base_train, base_test, mix_train, mix_test, y_tr,
                               n_latent_svd=12, n_latent_nmf=8)
    print(f"MCM dims added: {mcm_k}")

    y_t = target_transform(y_tr)
    target_mu = y_t.mean(axis=0).astype(np.float32)
    target_sd = (y_t.std(axis=0) + 1e-6).astype(np.float32)

    scenarios = np.array([s.scenario_id for s in base_train])
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros((len(base_train), 2), dtype=np.float32)
    oof_count = np.zeros(len(base_train), dtype=np.int32)
    test_preds_sum = np.zeros((len(base_test), 2), dtype=np.float32)
    n_models = 0
    val_summary = []

    device = torch.device(args.device)
    n_components = len(comp_vocab); n_types = len(type_vocab); n_props = len(TOP_PROPERTIES)

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(scenarios)):
        # Rebuild samples (fresh copies so we can mutate globals per-fold).
        train_copy = [_clone_sample(s) for s in base_train]
        test_copy = [_clone_sample(s) for s in base_test]
        # Attach anchors using only tr_idx as the candidate pool.
        added = attach_anchors(train_copy, test_copy, mix_train, mix_test, y_tr,
                               candidate_indices=tr_idx, k=args.k, mass_round=1)
        print(f"[fold {fold_idx+1}] anchors attached, +{added} dims -> globals dim "
              f"{train_copy[0].globals.shape[0]}")

        tr_samples = [train_copy[i] for i in tr_idx]
        va_samples = [train_copy[i] for i in va_idx]
        # test preds reuse test_copy

        global_dim_actual = int(tr_samples[0].globals.shape[0])
        G = np.stack([s.globals for s in tr_samples], axis=0)
        global_mu = G.mean(axis=0).astype(np.float32)
        global_sd = (G.std(axis=0) + 1e-6).astype(np.float32)

        for seed in range(args.seeds):
            print(f"\n[v28 fold {fold_idx+1}/{args.n_folds} seed {seed}] "
                  f"(global_dim={global_dim_actual})")
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
            print(f"  val_mae={best_val:.4f}")
            val_summary.append(dict(fold=fold_idx, seed=seed, val_mae=best_val))
            ids_va, preds_va = predict(model, va_samples, target_mu, target_sd, device,
                                       global_mu=global_mu, global_sd=global_sd)
            for i, sid in enumerate(ids_va):
                j = int(np.where(scenarios == sid)[0][0])
                oof_preds[j] += preds_va[i]; oof_count[j] += 1
            ids_te, preds_te = predict(model, test_copy, target_mu, target_sd, device,
                                       global_mu=global_mu, global_sd=global_sd)
            test_preds_sum += preds_te
            n_models += 1

    oof_avg = oof_preds / np.clip(oof_count, 1, None)[:, None]
    err = oof_avg - y_tr
    mv = float(np.mean(np.abs(err[:, 0]))); mo = float(np.mean(np.abs(err[:, 1])))
    std_v = y_tr[:, 0].std(); std_o = y_tr[:, 1].std()
    print(f"\n=== v28 OOF: visc {mv:.3f}, ox {mo:.3f}, norm {mv/std_v/2 + mo/std_o/2:.4f} ===")

    pd.DataFrame({
        "scenario_id": scenarios,
        "oof_viscosity": oof_avg[:, 0], "oof_oxidation": oof_avg[:, 1],
        "true_viscosity": y_tr[:, 0], "true_oxidation": y_tr[:, 1],
    }).to_csv(out_dir / "oof_predictions.csv", index=False)
    test_preds = test_preds_sum / n_models
    pd.DataFrame({
        "scenario_id": [s.scenario_id for s in base_test],
        "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %": test_preds[:, 0],
        "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm": test_preds[:, 1],
    }).to_csv(out_dir / "predictions_raw_headers.csv", index=False)
    with open(out_dir / "val_summary.json", "w") as f:
        json.dump(val_summary, f, indent=2)
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
