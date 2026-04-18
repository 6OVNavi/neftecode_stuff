"""Rich tabular feature build: 91 original + MCM scenario embeds + ratios/interactions.

Features added on top of build_tabular():
- MCM scenario embedding (21 dims: 12 SVD + 8 NMF + 1 freq)
- Physics ratio features: P/Zn, Ca/Mg, soap/base, biofuel*time, biofuel*time/T
- Log-sqrt transforms of concentration aggregates
- Type presence crosses (Mo×ZDDP present AND biofuel>0, etc.)
- Scenario-level property variance (heterogeneity measure)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import (
    CONDITION_DIM, COL_COMP, TOP_PROPERTIES, COMPONENT_TYPES,
    build_component_property_table, build_scenario_samples, build_vocabs,
    load_properties,
)
from .tabular import build_tabular, TYPE_SPECIFIC, TYPE_FEATURE_COUNT
from .mcm_features import compute_mcm_features


def build_tabular_rich(samples, mix_train: pd.DataFrame, mix_test: pd.DataFrame,
                       y_train: np.ndarray):
    """Return X (n, D_rich), feature_names, y, scenario_ids."""
    X_base, names_base, y, ids = build_tabular(samples)
    name_idx = {n: i for i, n in enumerate(names_base)}

    # 1) MCM scenario embeddings
    comp_feats, c2i, _ = compute_mcm_features(mix_train, mix_test, y_train,
                                              n_latent_svd=12, n_latent_nmf=8)
    K = comp_feats.shape[1]
    # Build per-sample scenario embedding from (mass, comp) lookup. Since we
    # already have samples with comp_ids (via vocab), we need a mapping from
    # vocab index → original component name. Easier: recompute from raw mix_df.
    mix_all = {sid: g for sid, g in pd.concat([mix_train, mix_test]).groupby('scenario_id', sort=False)}
    sce_emb = np.zeros((len(samples), K), dtype=np.float32)
    for i, s in enumerate(samples):
        g = mix_all.get(s.scenario_id)
        if g is None: continue
        v = np.zeros(K, dtype=np.float32); total = 0.0
        for comp, mass in zip(g['Компонент'], g['Массовая доля, %']):
            if comp in c2i:
                v += float(mass) * comp_feats[c2i[comp]]
                total += float(mass)
        if total > 0: v /= total
        sce_emb[i] = v

    # Standardize vs train stats (we get train slice from samples whose targets not None)
    train_mask = np.array([s.targets is not None for s in samples])
    if train_mask.any():
        mu = sce_emb[train_mask].mean(axis=0, keepdims=True)
        sd = sce_emb[train_mask].std(axis=0, keepdims=True) + 1e-6
        sce_emb = (sce_emb - mu) / sd

    # 2) Ratio / interaction features built from X_base columns
    def col(name): return X_base[:, name_idx[name]]

    extras = {}
    # Physics ratios (with safeguard against div-by-zero)
    def safe_div(a, b): return a / (np.abs(b) + 1e-6)
    extras['ratio_P_Zn'] = safe_div(col('agg_total_P'), col('agg_total_Zn') + 1.0)
    extras['ratio_Ca_Mg_denom_from_feat']= col('agg_total_Ca')  # placeholder (no Mg)
    extras['bio_x_time']  = col('bio_raw') * col('t_raw') / 100.0
    extras['bio_x_time_over_T'] = col('bio_raw') * col('t_raw') / (col('T_raw') + 1.0)
    extras['bio_sqrt_time'] = col('bio_raw') * np.sqrt(np.maximum(col('t_raw'), 0))
    extras['peroxide_factor_log'] = np.log1p(col('bio_raw') * col('t_raw') / (col('T_raw') + 273))
    # Mo × ZDDP × biofuel (synergy under biofuel stress)
    mo_mass = col('type_mass_Соединение_молибдена')
    zddp_mass = col('type_mass_Противоизносная_присадка')
    ao_mass = col('type_mass_Антиоксидант')
    extras['MoZDDPbio'] = mo_mass * zddp_mass * col('bio_raw')
    extras['MoAObio']   = mo_mass * ao_mass * col('bio_raw')
    extras['ZDDPbio'] = zddp_mass * col('bio_raw')
    # Interaction: high temp × loose viscosity modifier (viscosity loss risk)
    thickener_mass = col('type_mass_Загуститель')
    extras['thickener_high_T'] = thickener_mass * (col('T_raw') > 155).astype(float)
    # Missingness indicator: how many "new in train" components in test (relevant for novelty)
    extras['n_new_flag'] = (col('n_new') > 0).astype(float)
    # Log/sqrt transforms of concentration aggregates
    for c in ['agg_total_P','agg_total_Ca','agg_total_Zn','agg_total_S','agg_total_TBN',
              'agg_total_N','agg_total_B','agg_total_Mo','agg_total_NOACK']:
        if c in name_idx:
            extras[f'{c}_log'] = np.log1p(np.abs(col(c)))
            extras[f'{c}_sqrt'] = np.sqrt(np.maximum(col(c), 0))
    # biofuel vs sqrt_VI (hypothesis: higher VI reduces biofuel damage)
    if 'agg_mean_VI' in name_idx:
        extras['bio_over_VI'] = safe_div(col('bio_raw'), col('agg_mean_VI') + 100)

    ex_names = list(extras.keys())
    X_extra = np.stack([extras[k] for k in ex_names], axis=1)

    # Combine
    X_full = np.hstack([X_base, X_extra, sce_emb]).astype(np.float32)
    names = names_base + ex_names + [f'mcm_{i}' for i in range(sce_emb.shape[1])]
    return X_full, names, y, ids
