"""Train LubriSet with K-fold scenario-grouped CV + deep ensemble.

Run:
    python -m src.train --data_dir . --out_dir artifacts
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from .data import (
    CONDITION_DIM,
    COL_COMP,
    TOP_PROPERTIES,
    GLOBAL_FEAT_DIM,
    ScenarioSample,
    build_component_property_table,
    build_scenario_samples,
    build_vocabs,
    load_properties,
    target_transform,
    target_inverse_transform,
)

from .model import LubriSet


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class SetDataset(Dataset):
    def __init__(self, samples: list[ScenarioSample], target_mu=None, target_sd=None):
        self.samples = samples
        self.target_mu = target_mu
        self.target_sd = target_sd

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        item = {
            "comp_ids": s.comp_ids,
            "type_ids": s.type_ids,
            "props": s.props,
            "miss": s.miss_mask,
            "mass": s.mass,
            "is_new": s.is_new,
            "conditions": s.conditions,
            "globals": s.globals,
            "weight": float(s.weight),
        }
        if s.targets is not None:
            y_t = target_transform(s.targets[None, :])[0]
            if self.target_mu is not None:
                y_t = (y_t - self.target_mu) / self.target_sd
            item["target"] = y_t.astype(np.float32)
        item["scenario_id"] = s.scenario_id
        return item


def collate(batch, pad_n=None):
    B = len(batch)
    max_n = max(b["comp_ids"].shape[0] for b in batch)
    if pad_n:
        max_n = max(max_n, pad_n)
    P = batch[0]["props"].shape[1]
    comp_ids = np.zeros((B, max_n), dtype=np.int64)
    type_ids = np.zeros((B, max_n), dtype=np.int64)
    props = np.zeros((B, max_n, P), dtype=np.float32)
    miss = np.ones((B, max_n, P), dtype=np.float32)
    mass = np.zeros((B, max_n), dtype=np.float32)
    is_new = np.zeros((B, max_n), dtype=np.float32)
    conditions = np.zeros((B, batch[0]["conditions"].shape[0]), dtype=np.float32)
    globals_ = np.zeros((B, batch[0]["globals"].shape[0]), dtype=np.float32)
    weights = np.ones((B,), dtype=np.float32)
    pad_mask = np.ones((B, max_n), dtype=bool)  # True = pad
    has_target = "target" in batch[0]
    if has_target:
        tdim = batch[0]["target"].shape[0]
        targets = np.zeros((B, tdim), dtype=np.float32)
    scen_ids = []
    for i, b in enumerate(batch):
        n = b["comp_ids"].shape[0]
        comp_ids[i, :n] = b["comp_ids"]
        type_ids[i, :n] = b["type_ids"]
        props[i, :n] = b["props"]
        miss[i, :n] = b["miss"]
        mass[i, :n] = b["mass"]
        is_new[i, :n] = b["is_new"]
        conditions[i] = b["conditions"]
        globals_[i] = b["globals"]
        weights[i] = b.get("weight", 1.0)
        pad_mask[i, :n] = False
        scen_ids.append(b["scenario_id"])
        if has_target:
            targets[i] = b["target"]
    out = {
        "comp_ids": torch.from_numpy(comp_ids),
        "type_ids": torch.from_numpy(type_ids),
        "props": torch.from_numpy(props),
        "miss": torch.from_numpy(miss),
        "mass": torch.from_numpy(mass),
        "is_new": torch.from_numpy(is_new),
        "conditions": torch.from_numpy(conditions),
        "globals": torch.from_numpy(globals_),
        "weight": torch.from_numpy(weights),
        "pad_mask": torch.from_numpy(pad_mask),
        "scenario_id": scen_ids,
    }
    if has_target:
        out["target"] = torch.from_numpy(targets)
    return out


def component_dropout(batch: dict, p: float):
    """Randomly remove each non-pad component with prob p (but never drop below 3)."""
    if p <= 0:
        return batch
    mass = batch["mass"]
    pad_mask = batch["pad_mask"]
    B, N = mass.shape
    drop = torch.rand(B, N, device=mass.device) < p
    # Count active per row and ensure at least 3 remain.
    active = (~pad_mask).float()
    new_pad = pad_mask | drop
    remain = (~new_pad).float().sum(dim=1)
    # For rows where remain < 3, undo the drop for those rows.
    bad = remain < 3
    new_pad[bad] = pad_mask[bad]
    new_mass = mass.masked_fill(new_pad, 0.0)
    # Renormalize mass so it still sums to ~1 over remaining components.
    s = new_mass.sum(dim=1, keepdim=True).clamp_min(1e-6)
    new_mass = new_mass / s
    batch = {**batch, "pad_mask": new_pad, "mass": new_mass}
    return batch


def train_fold(
    train_samples,
    val_samples,
    n_components,
    n_types,
    n_props,
    condition_dim,
    global_mu,
    global_sd,
    device,
    target_mu,
    target_sd,
    epochs: int = 400,
    lr: float = 1e-3,
    wd: float = 1e-4,
    batch_size: int = 16,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 3,
    dropout: float = 0.1,
    id_dropout: float = 0.25,
    comp_dropout: float = 0.15,
    seed: int = 0,
    verbose: bool = True,
):
    set_seed(seed)
    train_ds = SetDataset(train_samples, target_mu, target_sd)
    val_ds = SetDataset(val_samples, target_mu, target_sd)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False
    )

    model = LubriSet(
        n_components=n_components,
        n_types=n_types,
        n_props=n_props,
        condition_dim=condition_dim,
        global_dim=GLOBAL_FEAT_DIM,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_targets=2,
        dropout=dropout,
        id_dropout=id_dropout,
    ).to(device)
    g_mu = torch.from_numpy(global_mu).to(device)
    g_sd = torch.from_numpy(global_sd).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Per-target loss weight; experimented with upweighting oxidation, but that
    # hurt viscosity more than it helped. Keep equal weights.
    loss_weights = torch.tensor([1.0, 1.0], device=device)

    # SWA: maintain a running average of the top-K best-by-val checkpoints.
    swa_k = 8
    swa_bank: list[tuple[float, dict]] = []  # (val_mae, state_dict)

    best_val = float("inf")
    best_state = None
    patience = 60
    patience_left = patience
    history = []

    for ep in range(epochs):
        model.train()
        total = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = component_dropout({k: (v.to(device) if torch.is_tensor(v) else v)
                                       for k, v in batch.items()}, p=comp_dropout)
            y = batch["target"]
            g = (batch["globals"] - g_mu) / g_sd
            pred = model(
                batch["comp_ids"], batch["type_ids"], batch["props"], batch["miss"],
                batch["mass"], batch["is_new"], batch["conditions"], batch["pad_mask"],
                global_feats=g,
            )
            err = F.smooth_l1_loss(pred, y, reduction="none")
            # Per-sample weight (1.0 for real train, <1 for pseudo-labels).
            sw = batch["weight"].unsqueeze(-1)
            loss = (err * loss_weights * sw).sum() / (sw.sum() * 2 + 1e-9)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            n_batches += 1
        sched.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch_t = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                g = (batch_t["globals"] - g_mu) / g_sd
                pred = model(
                    batch_t["comp_ids"], batch_t["type_ids"], batch_t["props"], batch_t["miss"],
                    batch_t["mass"], batch_t["is_new"], batch_t["conditions"], batch_t["pad_mask"],
                    global_feats=g,
                )
                val_preds.append(pred.cpu().numpy())
                val_targets.append(batch_t["target"].cpu().numpy())
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_mae = float(np.mean(np.abs(val_preds - val_targets)))
        history.append({"epoch": ep, "train_loss": total / max(n_batches, 1), "val_mae": val_mae})

        # Maintain SWA bank: keep top-K state dicts by val_mae.
        state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        swa_bank.append((val_mae, state_cpu))
        swa_bank.sort(key=lambda x: x[0])
        if len(swa_bank) > swa_k:
            swa_bank.pop()

        if val_mae < best_val:
            best_val = val_mae
            best_state = state_cpu
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
        if verbose and ep % 25 == 0:
            print(f"    ep={ep:3d} train={total/max(n_batches,1):.4f} val_mae={val_mae:.4f} best={best_val:.4f}")

    # Build SWA-averaged state from top-K bank.
    swa_state = {k: torch.zeros_like(v) for k, v in swa_bank[0][1].items()}
    for _, s in swa_bank:
        for k, v in s.items():
            swa_state[k] += v.float()
    for k in swa_state:
        swa_state[k] = (swa_state[k] / len(swa_bank)).to(swa_bank[0][1][k].dtype)

    # Evaluate SWA on val to decide whether to use it.
    model.load_state_dict(swa_state)
    model.eval()
    val_preds2, val_targets2 = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch_t = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            g = (batch_t["globals"] - g_mu) / g_sd
            pred = model(
                batch_t["comp_ids"], batch_t["type_ids"], batch_t["props"], batch_t["miss"],
                batch_t["mass"], batch_t["is_new"], batch_t["conditions"], batch_t["pad_mask"],
                global_feats=g,
            )
            val_preds2.append(pred.cpu().numpy())
            val_targets2.append(batch_t["target"].cpu().numpy())
    swa_mae = float(np.mean(np.abs(np.concatenate(val_preds2) - np.concatenate(val_targets2))))
    if swa_mae <= best_val:
        if verbose:
            print(f"    [SWA] {swa_mae:.4f} <= best {best_val:.4f}, using SWA")
        best_val = swa_mae
    else:
        if verbose:
            print(f"    [SWA] {swa_mae:.4f} > best {best_val:.4f}, keeping best")
        model.load_state_dict(best_state)
    return model, best_val, history


def predict(model, samples, target_mu, target_sd, device,
            global_mu=None, global_sd=None, batch_size=32, tta: int = 1):
    """Predict with optional test-time augmentation via random component permutation.

    Since the set is permutation-invariant by design, TTA only helps via dropout
    variance. Keep default tta=1 for clean single-pass; tta>1 activates MC dropout.
    """
    ds = SetDataset(samples)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False)
    g_mu = torch.from_numpy(global_mu).to(device) if global_mu is not None else None
    g_sd = torch.from_numpy(global_sd).to(device) if global_sd is not None else None

    all_runs = []
    ids = []
    for run in range(max(1, tta)):
        # Enable dropout sampling if tta > 1.
        if tta > 1:
            model.train()  # enable dropout
            for m in model.modules():
                # Keep BN/LN in eval mode.
                if isinstance(m, (nn.LayerNorm,)):
                    m.eval()
        else:
            model.eval()
        outs = []
        ids_run = []
        with torch.no_grad():
            for batch in dl:
                batch_t = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                g = None
                if g_mu is not None:
                    g = (batch_t["globals"] - g_mu) / g_sd
                pred = model(
                    batch_t["comp_ids"], batch_t["type_ids"], batch_t["props"], batch_t["miss"],
                    batch_t["mass"], batch_t["is_new"], batch_t["conditions"], batch_t["pad_mask"],
                    global_feats=g,
                )
                outs.append(pred.cpu().numpy())
                ids_run.extend(batch_t["scenario_id"])
        all_runs.append(np.concatenate(outs))
        if not ids:
            ids = ids_run
    preds = np.mean(np.stack(all_runs, axis=0), axis=0)
    # Un-standardize (target space).
    preds = preds * target_sd + target_mu
    preds_raw = target_inverse_transform(preds)
    return ids, preds_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pseudo_csv", default=None,
                    help="Path to a CSV with pseudo-labeled test scenarios (same"
                         " format as predictions.csv). These are added to the"
                         " training pool with sample_weight (see --pseudo_weight)"
                         " and are never put into the validation fold.")
    ap.add_argument("--pseudo_weight", type=float, default=0.5,
                    help="Weight applied to pseudo-labeled loss terms")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mix_train = pd.read_csv(data_dir / "daimler_mixtures_train.csv")
    mix_test = pd.read_csv(data_dir / "daimler_mixtures_test.csv")
    pr = load_properties(str(data_dir / "daimler_component_properties.csv"))
    wide_batch, wide_comp, mu, sd = build_component_property_table(pr)
    comp_vocab, type_vocab = build_vocabs(mix_train, mix_test)

    train_comp_set = set(mix_train[COL_COMP].unique())
    train_samples_all = build_scenario_samples(
        mix_train, wide_batch, wide_comp, mu, sd, comp_vocab, type_vocab,
        train_comp_set=train_comp_set, is_train=True,
    )
    test_samples = build_scenario_samples(
        mix_test, wide_batch, wide_comp, mu, sd, comp_vocab, type_vocab,
        train_comp_set=train_comp_set, is_train=False,
    )

    pseudo_samples: list = []
    if args.pseudo_csv:
        import numpy as _np
        pc = pd.read_csv(args.pseudo_csv)
        # Normalize column names: tolerate both 'scenario_id' + long target names,
        # or any 3-col csv where col1/2 are target_viscosity/target_oxidation.
        pc = pc.rename(columns={pc.columns[1]: "target_viscosity",
                                pc.columns[2]: "target_oxidation"})
        by_id = {r["scenario_id"]: (r["target_viscosity"], r["target_oxidation"])
                 for _, r in pc.iterrows()}
        for ts in test_samples:
            if ts.scenario_id not in by_id:
                continue
            y1, y2 = by_id[ts.scenario_id]
            ps = ScenarioSample(
                scenario_id=ts.scenario_id, comp_ids=ts.comp_ids, type_ids=ts.type_ids,
                props=ts.props, miss_mask=ts.miss_mask, mass=ts.mass,
                conditions=ts.conditions, is_new=ts.is_new, globals=ts.globals,
                targets=_np.array([y1, y2], dtype=_np.float32),
                weight=float(args.pseudo_weight),
            )
            pseudo_samples.append(ps)
        print(f"Loaded {len(pseudo_samples)} pseudo-labeled test scenarios "
              f"(weight={args.pseudo_weight})")

    # Target standardization based on whole train (in transformed space).
    y_raw = np.stack([s.targets for s in train_samples_all], axis=0)
    y_t = target_transform(y_raw)
    target_mu = y_t.mean(axis=0).astype(np.float32)
    target_sd = y_t.std(axis=0).astype(np.float32) + 1e-6
    print("Target transformed stats (mean, sd):", target_mu, target_sd)

    # Global feature normalization (fit on train only).
    G = np.stack([s.globals for s in train_samples_all], axis=0)
    global_mu = G.mean(axis=0).astype(np.float32)
    global_sd = (G.std(axis=0) + 1e-6).astype(np.float32)
    print(f"Global features: dim={G.shape[1]}")

    device = torch.device(args.device)
    n_props = len(TOP_PROPERTIES)
    n_components = len(comp_vocab)
    n_types = len(type_vocab)
    print(f"Components={n_components}, Types={n_types}, Props={n_props}")
    print(f"Train scenarios={len(train_samples_all)}, Test scenarios={len(test_samples)}")

    # K-fold ensemble.
    scenarios = np.array([s.scenario_id for s in train_samples_all])
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros((len(train_samples_all), 2), dtype=np.float32)
    oof_count = np.zeros(len(train_samples_all), dtype=np.int32)
    test_preds_sum = np.zeros((len(test_samples), 2), dtype=np.float32)
    n_models = 0

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(scenarios)):
        tr_samples = [train_samples_all[i] for i in tr_idx]
        va_samples = [train_samples_all[i] for i in va_idx]
        # Pseudo-labeled test scenarios: always in the training pool, never in val.
        if pseudo_samples:
            tr_samples = tr_samples + pseudo_samples
        for seed in range(args.n_seeds):
            print(f"\n[Fold {fold_idx+1}/{args.n_folds}, seed {seed}]")
            model, best_val, hist = train_fold(
                tr_samples, va_samples,
                n_components=n_components, n_types=n_types, n_props=n_props,
                condition_dim=CONDITION_DIM,
                global_mu=global_mu, global_sd=global_sd, device=device,
                target_mu=target_mu, target_sd=target_sd,
                epochs=args.epochs, batch_size=args.batch_size,
                d_model=args.d_model, n_layers=args.n_layers,
                seed=seed,
            )
            print(f"  -> best val_mae (normalized transformed) = {best_val:.4f}")

            # OOF predictions (in raw target space).
            ids_va, preds_va = predict(model, va_samples, target_mu, target_sd, device,
                                       global_mu=global_mu, global_sd=global_sd)
            for i, sid in enumerate(ids_va):
                j = int(np.where(scenarios == sid)[0][0])
                oof_preds[j] += preds_va[i]
                oof_count[j] += 1
            # Test predictions.
            ids_te, preds_te = predict(model, test_samples, target_mu, target_sd, device,
                                       global_mu=global_mu, global_sd=global_sd)
            test_preds_sum += preds_te
            n_models += 1

            # Save weights.
            torch.save(
                {"state_dict": model.state_dict(),
                 "target_mu": target_mu, "target_sd": target_sd,
                 "global_mu": global_mu, "global_sd": global_sd,
                 "comp_vocab": comp_vocab, "type_vocab": type_vocab,
                 "mu": mu, "sd": sd,
                 "config": {
                     "d_model": args.d_model, "n_layers": args.n_layers,
                     "n_components": n_components, "n_types": n_types, "n_props": n_props,
                     "condition_dim": CONDITION_DIM,
                     "global_dim": GLOBAL_FEAT_DIM,
                 }},
                out_dir / f"model_fold{fold_idx}_seed{seed}.pt",
            )

    # OOF report in raw target space.
    oof_avg = oof_preds / np.clip(oof_count, 1, None)[:, None]
    oof_err = oof_avg - y_raw
    print("\n=== OOF metrics (raw target space) ===")
    print(f"Viscosity MAE: {np.mean(np.abs(oof_err[:, 0])):.3f} (target range {y_raw[:,0].min():.1f}..{y_raw[:,0].max():.1f})")
    print(f"Oxidation MAE: {np.mean(np.abs(oof_err[:, 1])):.3f} (target range {y_raw[:,1].min():.1f}..{y_raw[:,1].max():.1f})")

    # Save OOF for analysis.
    oof_df = pd.DataFrame({
        "scenario_id": scenarios,
        "oof_viscosity": oof_avg[:, 0],
        "oof_oxidation": oof_avg[:, 1],
        "true_viscosity": y_raw[:, 0],
        "true_oxidation": y_raw[:, 1],
    })
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    # Ensemble test predictions.
    test_preds = test_preds_sum / n_models
    sids_te = [s.scenario_id for s in test_samples]
    sub = pd.DataFrame({
        "scenario_id": sids_te,
        "target_viscosity": test_preds[:, 0],
        "target_oxidation": test_preds[:, 1],
    })
    # The platform expects original raw column names; produce both canonical + raw-named files.
    sub.to_csv(out_dir / "predictions.csv", index=False)

    # Rename with raw target headers for submission.
    sub_raw = sub.rename(columns={
        "target_viscosity": "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
        "target_oxidation": "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm",
    })
    sub_raw.to_csv(out_dir / "predictions_raw_headers.csv", index=False)

    print(f"\nSaved {n_models} models, OOF file, and predictions.csv to {out_dir}/")
    meta = {
        "n_models": n_models, "n_folds": args.n_folds, "n_seeds": args.n_seeds,
        "oof_mae_visc": float(np.mean(np.abs(oof_err[:, 0]))),
        "oof_mae_ox": float(np.mean(np.abs(oof_err[:, 1]))),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
