"""FT-Transformer for tabular data (Gorishniy et al., 2021).

Each numeric feature -> (value * weight_i + bias_i) -> d_model embedding.
Add [CLS] token, apply Transformer encoder, predict from CLS.

Trained here with 5-fold CV x n_seeds, asinh target, SWA, pseudo-labels.
"""
from __future__ import annotations

import argparse
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
from .tabular import build_tabular


class FeatureTokenizer(nn.Module):
    """Numeric tokenizer: each feature -> d-dim token via (x * W_i + b_i)."""
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_features, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) -> (B, F, d)
        return x.unsqueeze(-1) * self.W + self.b


class FTTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 48, n_heads: int = 4,
                 n_layers: int = 3, n_targets: int = 2, dropout: float = 0.15):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           dropout=dropout, batch_first=True, norm_first=True,
                                           activation='gelu')
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(d_model, 1))
            for _ in range(n_targets)
        ])
        self.n_targets = n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        tokens = self.tokenizer(x)  # (B, F, d)
        cls = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        h = self.enc(tokens)
        h = self.norm(h[:, 0])  # CLS
        out = torch.stack([head(h).squeeze(-1) for head in self.heads], dim=-1)
        return out


def build_data():
    mix_train = pd.read_csv("daimler_mixtures_train.csv")
    mix_test = pd.read_csv("daimler_mixtures_test.csv")
    pr = load_properties("daimler_component_properties.csv")
    wb, wc, mu, sd = build_component_property_table(pr)
    cv, tv = build_vocabs(mix_train, mix_test)
    tr_s = build_scenario_samples(mix_train, wb, wc, mu, sd, cv, tv,
                                  train_comp_set=set(mix_train[COL_COMP].unique()),
                                  is_train=True)
    te_s = build_scenario_samples(mix_test, wb, wc, mu, sd, cv, tv,
                                  train_comp_set=set(mix_train[COL_COMP].unique()),
                                  is_train=False)
    Xtr, names, ytr, ids_tr = build_tabular(tr_s)
    Xte, _, _, ids_te = build_tabular(te_s)
    return Xtr, ytr, ids_tr, Xte, ids_te, names


def train_fold_ft(X_tr, y_tr, X_va, y_va, d_model=48, n_layers=3, dropout=0.15,
                  epochs=500, lr=3e-4, wd=1e-4, batch_size=32, seed=0,
                  pseudo_X=None, pseudo_y=None, pseudo_w=0.5):
    torch.manual_seed(seed); np.random.seed(seed)
    model = FTTransformer(X_tr.shape[1], d_model=d_model, n_layers=n_layers, dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # Prepare tensors
    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr).float()
    X_va_t = torch.from_numpy(X_va).float()
    y_va_t = torch.from_numpy(y_va).float()
    if pseudo_X is not None:
        X_aug = np.concatenate([X_tr, pseudo_X], axis=0)
        y_aug = np.concatenate([y_tr, pseudo_y], axis=0)
        w_aug = np.concatenate([np.ones(len(X_tr)), np.full(len(pseudo_X), pseudo_w)])
    else:
        X_aug = X_tr; y_aug = y_tr
        w_aug = np.ones(len(X_tr))
    X_aug_t = torch.from_numpy(X_aug).float()
    y_aug_t = torch.from_numpy(y_aug).float()
    w_aug_t = torch.from_numpy(w_aug).float()

    best_val = float('inf'); best_state = None; patience = 60; pat = patience
    swa_bank = []
    n_samples = len(X_aug_t)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_samples)
        total = 0.0; nb = 0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            pred = model(X_aug_t[idx])
            err = F.smooth_l1_loss(pred, y_aug_t[idx], reduction='none')
            w = w_aug_t[idx].unsqueeze(-1)
            loss = (err * w).sum() / (w.sum() * 2 + 1e-9)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            pred_va = model(X_va_t).numpy()
        val_mae = float(np.mean(np.abs(pred_va - y_va)))
        state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        swa_bank.append((val_mae, state_cpu))
        swa_bank.sort(key=lambda x: x[0]); swa_bank = swa_bank[:8]
        if val_mae < best_val:
            best_val = val_mae; best_state = state_cpu; pat = patience
        else:
            pat -= 1
            if pat <= 0: break

    # Build SWA
    swa_state = {k: torch.zeros_like(v) for k, v in swa_bank[0][1].items()}
    for _, s in swa_bank:
        for k, v in s.items(): swa_state[k] += v.float()
    for k in swa_state:
        swa_state[k] = (swa_state[k] / len(swa_bank)).to(swa_bank[0][1][k].dtype)
    # Evaluate SWA
    model.load_state_dict(swa_state); model.eval()
    with torch.no_grad():
        swa_pred = model(X_va_t).numpy()
    swa_mae = float(np.mean(np.abs(swa_pred - y_va)))
    if swa_mae <= best_val:
        return model, swa_mae
    model.load_state_dict(best_state)
    return model, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--pseudo_csv", default=None)
    ap.add_argument("--pseudo_weight", type=float, default=0.5)
    ap.add_argument("--out", default="/tmp/ft_oof.npy")
    ap.add_argument("--out_test", default="/tmp/ft_test.npy")
    args = ap.parse_args()

    Xtr, ytr, ids_tr, Xte, ids_te, names = build_data()
    scaler = StandardScaler(); Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
    y_t = target_transform(ytr)
    target_mu = y_t.mean(axis=0).astype(np.float32)
    target_sd = (y_t.std(axis=0) + 1e-6).astype(np.float32)
    y_norm = (y_t - target_mu) / target_sd

    pseudo_X = pseudo_y = None
    if args.pseudo_csv:
        pc = pd.read_csv(args.pseudo_csv)
        pc = pc.rename(columns={pc.columns[1]: 'v', pc.columns[2]: 'o'})
        by_id = {r.scenario_id: (r.v, r.o) for _, r in pc.iterrows()}
        pX = []; pY = []
        for i, sid in enumerate(ids_te):
            if sid in by_id:
                pX.append(Xte_s[i])
                y_raw = np.array([by_id[sid]], dtype=np.float32)
                y_tr_ = target_transform(y_raw)[0]
                pY.append((y_tr_ - target_mu) / target_sd)
        pseudo_X = np.stack(pX); pseudo_y = np.stack(pY)
        print(f"Loaded {len(pseudo_X)} pseudo-labeled rows (weight={args.pseudo_weight})")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    oof = np.zeros_like(y_norm)
    oof_cnt = np.zeros(len(y_norm), dtype=int)
    test_preds_sum = np.zeros((len(Xte_s), 2), dtype=np.float32)
    n_models = 0
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(Xtr_s)):
        X_tr_f, y_tr_f = Xtr_s[tr_idx], y_norm[tr_idx]
        X_va_f, y_va_f = Xtr_s[va_idx], y_norm[va_idx]
        for seed in range(args.n_seeds):
            print(f"[Fold {fold_idx+1}/{args.n_folds} seed {seed}]", flush=True)
            model, best = train_fold_ft(X_tr_f, y_tr_f, X_va_f, y_va_f,
                                        epochs=args.epochs, seed=seed,
                                        pseudo_X=pseudo_X, pseudo_y=pseudo_y,
                                        pseudo_w=args.pseudo_weight)
            print(f"  val_mae={best:.4f}")
            model.eval()
            with torch.no_grad():
                pred_va = model(torch.from_numpy(X_va_f).float()).numpy()
                pred_te = model(torch.from_numpy(Xte_s).float()).numpy()
            oof[va_idx] += pred_va
            oof_cnt[va_idx] += 1
            test_preds_sum += pred_te
            n_models += 1
    oof = oof / oof_cnt[:, None]
    test_preds = test_preds_sum / n_models
    # Un-normalize
    oof_raw = target_inverse_transform(oof * target_sd + target_mu)
    test_raw = target_inverse_transform(test_preds * target_sd + target_mu)
    mv = np.mean(np.abs(oof_raw[:,0] - ytr[:,0]))
    mo = np.mean(np.abs(oof_raw[:,1] - ytr[:,1]))
    print(f"\nFT-Transformer OOF: visc {mv:.3f}, ox {mo:.3f}, "
          f"norm {mv/ytr[:,0].std()/2 + mo/ytr[:,1].std()/2:.4f}")
    np.save(args.out, oof_raw)
    np.save(args.out_test, test_raw)
    print(f"Saved {n_models} models' predictions to {args.out} and {args.out_test}")


if __name__ == "__main__":
    main()
