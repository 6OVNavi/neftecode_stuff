"""v39: Arrhenius consistency auxiliary loss.

Hard Arrhenius head (v38) was too restrictive: OOF = 0.36 vs v20 0.20.
Instead of constraining the architecture we keep the unchanged LubriSet and
add a soft auxiliary loss that enforces Arrhenius temperature + time
consistency on the same-composition train pairs.

For every in-batch pair (A, B) with same composition (Jaccard ≥ 0.95), the
auxiliary loss is
   L_arr = || log(EOT_A / EOT_B) - ( Ea · (1/T_B - 1/T_A) + log(t_A/t_B) ) ||²

where Ea is a single globally learnable parameter (optimised jointly), and
(T, t) come from the scenario conditions.  The main prediction loss is
unchanged, so if the physics prior is unhelpful the model can ignore it by
driving the pair residuals to zero in other ways.  Weight of L_arr is a
small constant tuned on OOF.

No external constants required (Ea learned; R folded into Ea).
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
from torch.utils.data import DataLoader, Dataset

from .data import (
    CONDITION_DIM, COL_COMP, TOP_PROPERTIES, GLOBAL_FEAT_DIM,
    ScenarioSample, build_component_property_table, build_scenario_samples,
    build_vocabs, load_properties, target_transform, target_inverse_transform,
)
from .augment_globals import attach_mcm_globals
from .train import SetDataset, collate, component_dropout, set_seed, predict
from .model import LubriSet


COL_TEMP = "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C"
COL_TIME = "Время испытания | - Daimler Oxidation Test (DOT), ч"
COL_BIO = "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс"


def comp_sig(mix_df, mass_round=1):
    out = {}
    for sid, grp in mix_df.groupby("scenario_id"):
        out[sid] = frozenset((row[COL_COMP], round(row["Массовая доля, %"], mass_round))
                             for _, row in grp.iterrows())
    return out


def build_same_comp_pairs(samples, mix_train, min_jac=0.95):
    """Return list of (i, j) ordered pairs within `samples` that share
    component-mass composition at least `min_jac` Jaccard."""
    sig = comp_sig(mix_train, 1)
    sigs = [sig[s.scenario_id] for s in samples]
    n = len(samples)
    pairs = []
    by_sig = {}
    for i, s in enumerate(sigs):
        by_sig.setdefault(s, []).append(i)
    for members in by_sig.values():
        if len(members) < 2: continue
        for i in members:
            for j in members:
                if i != j:
                    pairs.append((i, j))
    return pairs


def train_fold_arr(train_samples, val_samples, *,
                   n_components, n_types, n_props, condition_dim,
                   global_dim_actual, global_mu, global_sd, device,
                   target_mu, target_sd, mix_train,
                   epochs=200, lr=1e-3, wd=1e-4, batch_size=16,
                   d_model=160, n_heads=4, n_layers=3,
                   dropout=0.10, id_dropout=0.25, comp_dropout=0.15,
                   mass_aug=0.15, seed=0,
                   arr_weight=0.05, asinh_scale_ox=20.0):
    set_seed(seed)
    train_ds = SetDataset(train_samples, target_mu, target_sd)
    val_ds = SetDataset(val_samples, target_mu, target_sd)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate, drop_last=False)

    model = LubriSet(
        n_components=n_components, n_types=n_types, n_props=n_props,
        condition_dim=condition_dim, global_dim=global_dim_actual,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, n_targets=2,
        dropout=dropout, id_dropout=id_dropout,
    ).to(device)
    # Global learnable Ea (in Kelvin units of Ea/R).  Init at 11000 (~90 kJ/mol).
    ea_param = nn.Parameter(torch.tensor(11000.0, device=device))

    g_mu = torch.from_numpy(global_mu).to(device)
    g_sd = torch.from_numpy(global_sd).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + [ea_param], lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_weights = torch.tensor([1.0, 1.0], device=device)

    # Pair indices within TRAIN samples
    pairs = build_same_comp_pairs(train_samples, mix_train, min_jac=0.95)
    pairs_i = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device) if pairs else None
    pairs_j = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device) if pairs else None

    swa_k = 8; swa_bank = []
    best_val = float("inf"); patience = 60; pat = patience; best_state = None

    # Precompute conditions for pair loss (from samples, same ordering as train_ds)
    cond_tensor = torch.tensor(
        np.stack([s.conditions for s in train_samples], axis=0), device=device, dtype=torch.float32
    )

    for ep in range(epochs):
        model.train()
        total = 0.0; total_arr = 0.0; nb = 0
        for batch in train_loader:
            batch = component_dropout({k: (v.to(device) if torch.is_tensor(v) else v)
                                       for k, v in batch.items()},
                                      p=comp_dropout, mass_aug=mass_aug)
            y = batch["target"]
            g = (batch["globals"] - g_mu) / g_sd
            pred = model(
                batch["comp_ids"], batch["type_ids"], batch["props"], batch["miss"],
                batch["mass"], batch["is_new"], batch["conditions"], batch["pad_mask"],
                global_feats=g,
            )
            err = F.smooth_l1_loss(pred, y, reduction="none")
            sw = batch["weight"].unsqueeze(-1)
            loss_main = (err * loss_weights * sw).sum() / (sw.sum() * 2 + 1e-9)

            loss = loss_main

            # Arrhenius pair consistency: sample a random subset of pairs this step
            if pairs_i is not None and arr_weight > 0 and len(pairs) >= 4:
                n_sample = min(32, len(pairs))
                perm = torch.randperm(len(pairs), device=device)[:n_sample]
                pi = pairs_i[perm]; pj = pairs_j[perm]
                # Predict on ALL train samples at once via a dedicated mini-batch?
                # Too expensive per step — instead re-use the model on a small
                # extra pass with the relevant samples.
                idx_all = torch.unique(torch.cat([pi, pj])).tolist()
                sub_ds = SetDataset([train_samples[i] for i in idx_all],
                                    target_mu, target_sd)
                sub_batch = collate([sub_ds[k] for k in range(len(sub_ds))])
                sub_batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                             for k, v in sub_batch.items()}
                sub_g = (sub_batch["globals"] - g_mu) / g_sd
                sub_pred = model(
                    sub_batch["comp_ids"], sub_batch["type_ids"],
                    sub_batch["props"], sub_batch["miss"], sub_batch["mass"],
                    sub_batch["is_new"], sub_batch["conditions"], sub_batch["pad_mask"],
                    global_feats=sub_g,
                )
                # sub_pred has predictions in asinh space, index [:, 1] is ox
                # map indices back
                idx_map = {v: k for k, v in enumerate(idx_all)}
                pi_local = torch.tensor([idx_map[i.item()] for i in pi], device=device)
                pj_local = torch.tensor([idx_map[j.item()] for j in pj], device=device)
                tm1 = float(target_mu[1]); ts1 = float(target_sd[1])
                ox_i_asinh = sub_pred[pi_local, 1] * ts1 + tm1  # undo z-norm
                ox_j_asinh = sub_pred[pj_local, 1] * ts1 + tm1
                # Convert back to raw EOT via inverse asinh
                eot_i = torch.sinh(ox_i_asinh) * asinh_scale_ox
                eot_j = torch.sinh(ox_j_asinh) * asinh_scale_ox
                # guard against non-positive
                eot_i = torch.clamp(eot_i, min=0.5)
                eot_j = torch.clamp(eot_j, min=0.5)
                T_i_K = cond_tensor[pi, 0] + 273.15
                T_j_K = cond_tensor[pj, 0] + 273.15
                t_i = torch.clamp(cond_tensor[pi, 1], min=1.0)
                t_j = torch.clamp(cond_tensor[pj, 1], min=1.0)
                # Expected Arrhenius: log(eot_i / eot_j) = -Ea*(1/T_i - 1/T_j) + log(t_i/t_j)
                lhs = torch.log(eot_i) - torch.log(eot_j)
                rhs = -ea_param * (1.0 / T_i_K - 1.0 / T_j_K) + torch.log(t_i / t_j)
                loss_arr = F.smooth_l1_loss(lhs, rhs)
                loss = loss + arr_weight * loss_arr
                total_arr += loss_arr.item()

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [ea_param], 1.0)
            opt.step()
            total += loss_main.item(); nb += 1
        sched.step()

        # val
        model.eval()
        val_preds = []; val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                g = (batch["globals"] - g_mu) / g_sd
                pred = model(
                    batch["comp_ids"], batch["type_ids"], batch["props"], batch["miss"],
                    batch["mass"], batch["is_new"], batch["conditions"], batch["pad_mask"],
                    global_feats=g,
                )
                val_preds.append(pred.cpu().numpy())
                val_targets.append(batch["target"].cpu().numpy())
        val_preds = np.concatenate(val_preds); val_targets = np.concatenate(val_targets)
        val_mae = np.mean(np.abs(val_preds - val_targets))

        if val_mae < best_val - 1e-5:
            best_val = val_mae; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat = patience
        else:
            pat -= 1
        # SWA bank
        swa_bank.append((val_mae, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
        swa_bank.sort(key=lambda x: x[0])
        swa_bank = swa_bank[:swa_k]

        if ep % 25 == 0:
            print(f"    ep={ep:3d} train={total/max(nb,1):.4f} arr={total_arr/max(nb,1):.4f} "
                  f"val_mae={val_mae:.4f} best={best_val:.4f} Ea/R={ea_param.item():.0f}", flush=True)
        if pat <= 0: break

    # SWA
    if len(swa_bank) >= 2:
        avg = {k: sum(s[1][k].float() for s in swa_bank) / len(swa_bank)
               for k in swa_bank[0][1].keys()}
        swa_model = LubriSet(
            n_components=n_components, n_types=n_types, n_props=n_props,
            condition_dim=condition_dim, global_dim=global_dim_actual,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, n_targets=2,
            dropout=0.0, id_dropout=0.0,
        ).to(device)
        swa_model.load_state_dict(avg)
        swa_model.eval()
        swa_val_preds = []; swa_val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                g = (batch["globals"] - g_mu) / g_sd
                pred = swa_model(
                    batch["comp_ids"], batch["type_ids"], batch["props"], batch["miss"],
                    batch["mass"], batch["is_new"], batch["conditions"], batch["pad_mask"],
                    global_feats=g,
                )
                swa_val_preds.append(pred.cpu().numpy())
                swa_val_targets.append(batch["target"].cpu().numpy())
        swa_val_preds = np.concatenate(swa_val_preds); swa_val_targets = np.concatenate(swa_val_targets)
        swa_val = np.mean(np.abs(swa_val_preds - swa_val_targets))
        if swa_val < best_val:
            print(f"    [SWA] {swa_val:.4f} <= best {best_val:.4f}, using SWA")
            return swa_model, float(swa_val), None
        else:
            print(f"    [SWA] {swa_val:.4f} > best {best_val:.4f}, keeping best")
    model.load_state_dict(best_state)
    return model, float(best_val), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="artifacts_v39")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arr_weight", type=float, default=0.05)
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
    target_mu_t = torch.from_numpy(target_mu)
    target_sd_t = torch.from_numpy(target_sd)

    scenarios = np.array([s.scenario_id for s in train_samples])
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(train_samples), 2), dtype=np.float32)
    oof_count = np.zeros(len(train_samples), dtype=np.int32)
    test_preds_sum = np.zeros((len(test_samples), 2), dtype=np.float32)
    n_models = 0
    device = torch.device(args.device)
    n_components = len(comp_vocab); n_types = len(type_vocab); n_props = len(TOP_PROPERTIES)

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(scenarios)):
        tr_samples = [train_samples[i] for i in tr_idx]
        va_samples = [train_samples[i] for i in va_idx]
        for seed in range(args.seeds):
            print(f"\n[v39 fold {fold_idx+1}/{args.n_folds} seed {seed}] arr_weight={args.arr_weight}", flush=True)
            model, best_val, _ = train_fold_arr(
                tr_samples, va_samples,
                n_components=n_components, n_types=n_types, n_props=n_props,
                condition_dim=CONDITION_DIM, global_dim_actual=global_dim_actual,
                global_mu=global_mu, global_sd=global_sd, device=device,
                target_mu=target_mu, target_sd=target_sd, mix_train=mix_train,
                epochs=args.epochs, batch_size=16,
                d_model=160, n_layers=3, dropout=0.10, comp_dropout=0.15,
                id_dropout=0.25, mass_aug=0.15, seed=seed * 100 + fold_idx,
                arr_weight=args.arr_weight,
            )
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
    print(f"\n=== v39 OOF: visc {mv:.3f}, ox {mo:.3f}, norm {mv/std_v/2 + mo/std_o/2:.4f} ===")

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
    print(f"Saved v39 to {out_dir}/")


if __name__ == "__main__":
    main()
