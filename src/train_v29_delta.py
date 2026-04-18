"""v29: conditional-delta model.

We have 47 train scenarios with same-composition siblings.  That gives us
~100 ordered (A, B) pairs where only conditions (T, t, bio, cat) differ.
Train a small regressor on (mixture_features, Δconds) → Δy.

At test time, for each test with a same-composition train anchor, we predict
Δy and add it to the anchor's target.  For tests without anchors we fall back
to the base Set Transformer ensemble.

This is ORTHOGONAL to the base model: it explicitly targets the conditions
dependence the base model is weak at, measured by v20 OOF MAE on same-comp
subset being v=68 o=8 (noise floor v=0.2 o=0.4).

We validate on OOF: for each train sample, we predict Δy using the best
same-comp sibling in the OTHER folds, and compare to the real Δy.  We also
blend with the base model's OOF prediction and report the best blend.
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

from .data import (
    COL_COMP, TOP_PROPERTIES, ScenarioSample,
    build_component_property_table, build_scenario_samples,
    build_vocabs, load_properties, target_transform, target_inverse_transform,
)
from .augment_globals import attach_mcm_globals


COL_TEMP = "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C"
COL_TIME = "Время испытания | - Daimler Oxidation Test (DOT), ч"
COL_BIO = "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс"
COL_CAT = "Дозировка катализатора, категория"
COL_VISC = "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %"
COL_OX = "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"


def comp_sig(df, mass_round=1):
    out = {}
    for sid, grp in df.groupby("scenario_id"):
        out[sid] = frozenset((row[COL_COMP], round(row["Массовая доля, %"], mass_round))
                             for _, row in grp.iterrows())
    return out


def scenario_feats(sample: ScenarioSample, n_types: int = 16) -> np.ndarray:
    """Scenario-level mixture representation: mass-weighted std props + type
    counts + pooled globals.  Invariant to component ordering."""
    obs = (1.0 - sample.miss_mask) * sample.props  # (n, P)
    w = sample.mass[:, None]
    pooled = (w * obs).sum(axis=0)  # (P,)
    # Mass-weighted std
    mean2 = pooled
    var = (w * (obs - mean2) ** 2).sum(axis=0)
    stdp = np.sqrt(np.clip(var, 0, None))
    type_mass = np.zeros(n_types, dtype=np.float32)
    for i, t in enumerate(sample.type_ids):
        ti = int(t)
        if ti < n_types:
            type_mass[ti] += float(sample.mass[i])
    return np.concatenate([pooled.astype(np.float32),
                           stdp.astype(np.float32),
                           type_mass,
                           sample.globals.astype(np.float32)]).astype(np.float32)


def build_pairs(samples, sigs, conds, y_raw, min_jac=0.95):
    """Return list of dicts: {i, j, feats_i, Δconds, Δy_raw}."""
    n = len(samples)
    # index by signature
    by_sig = {}
    for i, s in enumerate(samples):
        by_sig.setdefault(sigs[s.scenario_id], []).append(i)
    pairs = []
    for sig, members in by_sig.items():
        if len(members) < 2:
            # also include near-matches across sigs
            pass
        for i in members:
            for j in members:
                if i == j: continue
                ci = conds[samples[i].scenario_id]
                cj = conds[samples[j].scenario_id]
                delta_c = np.array([ci[0] - cj[0], ci[1] - cj[1],
                                    ci[2] - cj[2],
                                    0.0 if ci[3] == cj[3] else 1.0],
                                   dtype=np.float32)
                pairs.append(dict(i=i, j=j, delta_c=delta_c,
                                  delta_y=(y_raw[i] - y_raw[j]).astype(np.float32)))
    return pairs


class DeltaNet(nn.Module):
    def __init__(self, in_dim, cond_dim=4, hidden=128):
        super().__init__()
        self.mix_enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + cond_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, 2),
        )

    def forward(self, feats, dcond):
        h = self.mix_enc(feats)
        out = self.head(torch.cat([h, dcond], dim=-1))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="artifacts_v29")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--base", default="artifacts_v20",
                    help="Base model artifact dir for OOF + test predictions")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--blend", type=float, default=0.5,
                    help="Blend weight for delta correction vs base model")
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

    sigs_tr = comp_sig(mix_train, mass_round=1)
    sigs_te = comp_sig(mix_test, mass_round=1)

    def get_conds(df):
        out = {}
        for sid, grp in df.groupby("scenario_id"):
            r = grp.iloc[0]
            out[sid] = (float(r[COL_TEMP]), float(r[COL_TIME]),
                        float(r[COL_BIO]), float(r[COL_CAT]))
        return out
    conds_tr = get_conds(mix_train)
    conds_te = get_conds(mix_test)

    # Features per train sample
    Xtr = np.stack([scenario_feats(s) for s in train_samples], axis=0)
    feat_mu = Xtr.mean(0); feat_sd = Xtr.std(0) + 1e-6
    Xtr = ((Xtr - feat_mu) / feat_sd).astype(np.float32)
    Xte = np.stack([scenario_feats(s) for s in test_samples], axis=0)
    Xte = ((Xte - feat_mu) / feat_sd).astype(np.float32)

    # OOF-safe pairs construction: same-sig pairs that both fall within the
    # current training fold.
    scenarios = np.array([s.scenario_id for s in train_samples])
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    base_oof = pd.read_csv(Path(args.base) / "oof_predictions.csv")
    base_oof = base_oof.sort_values("scenario_id").set_index("scenario_id")
    base_test = pd.read_csv(Path(args.base) / "predictions_raw_headers.csv")
    base_test = base_test.sort_values("scenario_id").reset_index(drop=True)
    # re-index base oof to match our sample order
    base_oof_preds = np.stack(
        [base_oof.loc[sid][["oof_viscosity", "oof_oxidation"]].values for sid in scenarios]
    ).astype(np.float32)

    device = torch.device("cpu")
    all_delta_pred_oof = np.zeros((len(train_samples), 2), dtype=np.float32)
    oof_has_anchor = np.zeros(len(train_samples), dtype=bool)
    delta_test_sum = np.zeros((len(test_samples), 2), dtype=np.float32)
    test_has_anchor_count = np.zeros(len(test_samples), dtype=int)

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(scenarios)):
        pairs = build_pairs([train_samples[i] for i in tr_idx],
                            sigs_tr, conds_tr, y_tr[tr_idx], min_jac=0.95)
        if not pairs:
            print(f"[fold {fold_idx+1}] no same-sig pairs in tr -- skipping delta model")
            continue
        # Recover original indices (pair.i is local to tr slice -> map back)
        for p in pairs:
            p["i_orig"] = int(tr_idx[p["i"]])
            p["j_orig"] = int(tr_idx[p["j"]])
        print(f"[fold {fold_idx+1}] delta training pairs: {len(pairs)}")

        feats_i = torch.tensor(np.stack([Xtr[p["i_orig"]] for p in pairs]), dtype=torch.float32)
        dc = torch.tensor(np.stack([p["delta_c"] for p in pairs]), dtype=torch.float32)
        # target delta in asinh space for stability
        delta_y_at = np.stack([target_transform(y_tr[p["i_orig"]][None])[0]
                               - target_transform(y_tr[p["j_orig"]][None])[0]
                               for p in pairs]).astype(np.float32)
        ydt = torch.tensor(delta_y_at, dtype=torch.float32)

        model = DeltaNet(in_dim=Xtr.shape[1], hidden=args.hidden).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(len(pairs))
            bs = 64
            for b in range(0, len(pairs), bs):
                idx = perm[b:b+bs]
                pred = model(feats_i[idx], dc[idx])
                loss = F.smooth_l1_loss(pred, ydt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

        # Predict OOF: for each va_idx sample, find a same-sig anchor inside tr_idx
        model.eval()
        with torch.no_grad():
            for vi in va_idx:
                sid = train_samples[vi].scenario_id
                sig = sigs_tr[sid]
                anchors = [ti for ti in tr_idx if sigs_tr[train_samples[ti].scenario_id] == sig]
                if not anchors:
                    continue
                # pick the anchor with closest conditions to the target
                ci = conds_tr[sid]
                best = min(anchors, key=lambda a: abs(ci[0]-conds_tr[train_samples[a].scenario_id][0])/10
                                                + abs(ci[1]-conds_tr[train_samples[a].scenario_id][1])/48)
                feats = torch.tensor(Xtr[vi:vi+1], dtype=torch.float32)
                cb = conds_tr[train_samples[best].scenario_id]
                dc_t = torch.tensor([[ci[0]-cb[0], ci[1]-cb[1], ci[2]-cb[2],
                                      0.0 if ci[3] == cb[3] else 1.0]], dtype=torch.float32)
                dpred_at = model(feats, dc_t).cpu().numpy()[0]
                # add to anchor's asinh target and invert
                anchor_at = target_transform(y_tr[best][None])[0]
                predicted_at = anchor_at + dpred_at
                all_delta_pred_oof[vi] = target_inverse_transform(predicted_at[None])[0]
                oof_has_anchor[vi] = True

        # Predict test: for each test, find same-sig train anchor (any train)
        with torch.no_grad():
            for ti in range(len(test_samples)):
                tid = test_samples[ti].scenario_id
                sig = sigs_te[tid]
                anchors = [a for a, s in enumerate(train_samples) if sigs_tr[s.scenario_id] == sig]
                if not anchors:
                    continue
                cti = conds_te[tid]
                best = min(anchors, key=lambda a: abs(cti[0]-conds_tr[train_samples[a].scenario_id][0])/10
                                                 + abs(cti[1]-conds_tr[train_samples[a].scenario_id][1])/48)
                feats = torch.tensor(Xte[ti:ti+1], dtype=torch.float32)
                cb = conds_tr[train_samples[best].scenario_id]
                dc_t = torch.tensor([[cti[0]-cb[0], cti[1]-cb[1], cti[2]-cb[2],
                                      0.0 if cti[3] == cb[3] else 1.0]], dtype=torch.float32)
                dpred_at = model(feats, dc_t).cpu().numpy()[0]
                anchor_at = target_transform(y_tr[best][None])[0]
                predicted_at = anchor_at + dpred_at
                delta_test_sum[ti] += target_inverse_transform(predicted_at[None])[0]
                test_has_anchor_count[ti] += 1

    # Average delta test predictions over folds that had anchors
    delta_test = delta_test_sum / np.clip(test_has_anchor_count, 1, None)[:, None]
    has_anchor_test = test_has_anchor_count > 0

    # Validate delta approach on OOF
    y_raw = y_tr
    std_v, std_o = y_raw[:, 0].std(), y_raw[:, 1].std()

    def norm_score(pred, truth):
        mv = np.mean(np.abs(pred[:, 0] - truth[:, 0]))
        mo = np.mean(np.abs(pred[:, 1] - truth[:, 1]))
        return mv/std_v/2 + mo/std_o/2, mv, mo

    base_full, bv, bo = norm_score(base_oof_preds, y_raw)
    print(f"\nBase {args.base} OOF (full 167):     norm={base_full:.4f}  v={bv:.2f}  o={bo:.2f}")

    anchor_n = int(oof_has_anchor.sum())
    if anchor_n > 0:
        subset = oof_has_anchor
        base_sub, sv, so = norm_score(base_oof_preds[subset], y_raw[subset])
        delta_sub, dv, do = norm_score(all_delta_pred_oof[subset], y_raw[subset])
        print(f"Base on {anchor_n} anchored-OOF:  norm={base_sub:.4f}  v={sv:.2f}  o={so:.2f}")
        print(f"Delta on {anchor_n} anchored-OOF: norm={delta_sub:.4f}  v={dv:.2f}  o={do:.2f}")
        # Try multiple blend weights
        for w in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
            b = (1-w)*base_oof_preds[subset] + w*all_delta_pred_oof[subset]
            n, mv, mo = norm_score(b, y_raw[subset])
            print(f"  blend {w:.1f}: norm={n:.4f}  v={mv:.2f}  o={mo:.2f}")
        # Pick best blend
        best_blend = None; best_score = 1e9
        for w in np.linspace(0, 1, 21):
            b = (1-w)*base_oof_preds[subset] + w*all_delta_pred_oof[subset]
            n, _, _ = norm_score(b, y_raw[subset])
            if n < best_score:
                best_score = n; best_blend = w
        print(f"Best blend = {best_blend:.2f} with subset score {best_score:.4f}")

        # Final OOF: use best_blend where anchor exists, base elsewhere
        final = base_oof_preds.copy()
        final[subset] = (1-best_blend)*base_oof_preds[subset] + best_blend*all_delta_pred_oof[subset]
        ff, fv, fo = norm_score(final, y_raw)
        print(f"Combined OOF (anchors blended {best_blend:.2f}): norm={ff:.4f}  v={fv:.2f}  o={fo:.2f}")

        # Final test: same blend where anchor exists
        base_test_mat = base_test[[
            "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
            "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"
        ]].values.astype(np.float32)
        final_test = base_test_mat.copy()
        if has_anchor_test.any():
            final_test[has_anchor_test] = ((1-best_blend)*base_test_mat[has_anchor_test]
                                           + best_blend*delta_test[has_anchor_test])

        pd.DataFrame({
            "scenario_id": scenarios,
            "oof_viscosity": final[:, 0], "oof_oxidation": final[:, 1],
            "true_viscosity": y_raw[:, 0], "true_oxidation": y_raw[:, 1],
        }).to_csv(out_dir / "oof_predictions.csv", index=False)
        pd.DataFrame({
            "scenario_id": base_test["scenario_id"].values,
            "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %": final_test[:, 0],
            "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm": final_test[:, 1],
        }).to_csv(out_dir / "predictions_raw_headers.csv", index=False)
        with open(out_dir / "result.json", "w") as f:
            json.dump({"best_blend": float(best_blend),
                       "oof_score_full": float(ff),
                       "oof_score_base": float(base_full),
                       "n_anchored": int(anchor_n),
                       "n_test_anchored": int(has_anchor_test.sum())}, f, indent=2)
        print(f"\nSaved v29 to {out_dir}/")


if __name__ == "__main__":
    main()
