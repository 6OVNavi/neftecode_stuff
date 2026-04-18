"""Augment ScenarioSample.globals with MCM-derived scenario-level features.

Strategy: compute scenario embedding = sum(mass_i * comp_feats_i) using MCM
latents, append to each sample's globals vector. Standardized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .mcm_features import compute_mcm_features


def attach_mcm_globals(train_samples, test_samples,
                       mix_train: pd.DataFrame, mix_test: pd.DataFrame,
                       y_train: np.ndarray, n_latent_svd: int = 12,
                       n_latent_nmf: int = 8):
    """Compute scenario-level MCM embeddings and concat to .globals in-place.

    Returns the number of MCM dims added.
    """
    comp_feats, c2i, meta = compute_mcm_features(
        mix_train, mix_test, y_train,
        n_latent_svd=n_latent_svd, n_latent_nmf=n_latent_nmf,
    )
    K = comp_feats.shape[1]
    # For each scenario, compute weighted sum of its components' comp_feats.
    # We use mix_train / mix_test directly (raw mass fractions), since
    # sample.mass is already normalized per scenario.
    def scenario_emb(mix_df):
        embs = []
        sids = []
        for sid, g in mix_df.groupby('scenario_id', sort=False):
            v = np.zeros(K, dtype=np.float32)
            total = 0.0
            for comp, mass in zip(g['Компонент'], g['Массовая доля, %']):
                if comp in c2i:
                    v += float(mass) * comp_feats[c2i[comp]]
                    total += float(mass)
            if total > 0:
                v /= total
            embs.append(v)
            sids.append(sid)
        return np.stack(embs), sids

    tr_emb, tr_sids = scenario_emb(mix_train)
    te_emb, te_sids = scenario_emb(mix_test)

    # Standardize using train stats
    mu = tr_emb.mean(axis=0, keepdims=True)
    sd = tr_emb.std(axis=0, keepdims=True) + 1e-6
    tr_emb = (tr_emb - mu) / sd
    te_emb = (te_emb - mu) / sd

    tr_map = dict(zip(tr_sids, tr_emb))
    te_map = dict(zip(te_sids, te_emb))

    for s in train_samples:
        extra = tr_map.get(s.scenario_id)
        if extra is None:
            extra = np.zeros(K, dtype=np.float32)
        s.globals = np.concatenate([s.globals, extra.astype(np.float32)])
    for s in test_samples:
        extra = te_map.get(s.scenario_id)
        if extra is None:
            extra = np.zeros(K, dtype=np.float32)
        s.globals = np.concatenate([s.globals, extra.astype(np.float32)])
    return K
