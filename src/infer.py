"""Load trained ensemble and emit predictions.csv for the test set."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import (
    CONDITION_DIM,
    COL_COMP,
    TOP_PROPERTIES,
    build_component_property_table,
    build_scenario_samples,
    build_vocabs,
    load_properties,
    target_inverse_transform,
)
from .model import LubriSet
from .train import SetDataset, collate
from torch.utils.data import DataLoader


def load_model(ckpt_path: str, device: torch.device) -> tuple[LubriSet, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = LubriSet(
        n_components=cfg["n_components"], n_types=cfg["n_types"],
        n_props=cfg["n_props"], condition_dim=cfg["condition_dim"],
        global_dim=cfg.get("global_dim", 0),
        d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_targets=2,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    art_dir = Path(args.artifacts)
    device = torch.device(args.device)

    mix_train = pd.read_csv(data_dir / "daimler_mixtures_train.csv")
    mix_test = pd.read_csv(data_dir / "daimler_mixtures_test.csv")
    pr = load_properties(str(data_dir / "daimler_component_properties.csv"))
    wide_batch, wide_comp, mu, sd = build_component_property_table(pr)
    comp_vocab, type_vocab = build_vocabs(mix_train, mix_test)
    train_comp_set = set(mix_train[COL_COMP].unique())
    test_samples = build_scenario_samples(
        mix_test, wide_batch, wide_comp, mu, sd, comp_vocab, type_vocab,
        train_comp_set=train_comp_set, is_train=False,
    )

    ckpt_files = sorted(art_dir.glob("model_fold*_seed*.pt"))
    if not ckpt_files:
        raise SystemExit(f"No checkpoints in {art_dir}")

    all_preds = []
    target_mu = target_sd = None
    global_mu = global_sd = None
    for ck in ckpt_files:
        model, ck_meta = load_model(str(ck), device)
        if target_mu is None:
            target_mu = ck_meta["target_mu"]
            target_sd = ck_meta["target_sd"]
            global_mu = ck_meta.get("global_mu")
            global_sd = ck_meta.get("global_sd")
        g_mu = torch.from_numpy(global_mu).to(device) if global_mu is not None else None
        g_sd = torch.from_numpy(global_sd).to(device) if global_sd is not None else None
        ds = SetDataset(test_samples)
        dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate)
        outs, ids = [], []
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
                ids.extend(batch_t["scenario_id"])
        preds = np.concatenate(outs) * target_sd + target_mu
        preds_raw = target_inverse_transform(preds)
        all_preds.append(preds_raw)

    avg = np.mean(np.stack(all_preds, axis=0), axis=0)
    sub = pd.DataFrame({
        "scenario_id": ids,
        "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %": avg[:, 0],
        "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm": avg[:, 1],
    })
    sub.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(sub)} rows from {len(ckpt_files)} models")


if __name__ == "__main__":
    main()
