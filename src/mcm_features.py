"""MCM-style (Matrix Completion Method) pretraining for component embeddings.

Inspired by Jirasek et al. (2209.00605, 2001.10675): we learn per-component
latent vectors θ_i by factorizing observed target data across scenarios.

Two parallel factorizations:
- θ^(ℓ)_i: learn low-rank decomposition of (scenario × component) mass matrix
  via SVD/NMF → captures co-occurrence structure (which components go together).
- θ^(T)_i: learn per-component "effect size" per target via ridge regression:
  y ≈ M @ (θ per target), where M is (n_scenarios, n_components) mass matrix.
  Captures LINEAR contribution of each component to each target.

Output: per-component feature matrix augmenting the existing component properties.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.linear_model import Ridge


def build_component_ids(mix_train: pd.DataFrame, mix_test: pd.DataFrame):
    all_comps = sorted(set(mix_train['Компонент']).union(mix_test['Компонент']))
    comp_to_idx = {c: i for i, c in enumerate(all_comps)}
    return all_comps, comp_to_idx


def build_mass_matrix(mix_df: pd.DataFrame, comp_to_idx: dict) -> tuple[np.ndarray, list[str]]:
    """Return (n_scenarios, n_components) mass matrix + scenario_id list."""
    sids = []
    rows = []
    for sid, g in mix_df.groupby('scenario_id', sort=False):
        v = np.zeros(len(comp_to_idx), dtype=np.float32)
        for comp, mass in zip(g['Компонент'], g['Массовая доля, %']):
            if comp in comp_to_idx:
                v[comp_to_idx[comp]] += float(mass)
        total = v.sum()
        if total > 0:
            v = v / total
        sids.append(sid)
        rows.append(v)
    return np.stack(rows), sids


def compute_mcm_features(
    mix_train: pd.DataFrame, mix_test: pd.DataFrame,
    y_train: np.ndarray,
    n_latent_svd: int = 12,
    n_latent_nmf: int = 8,
    ridge_alpha: float = 1.0,
    use_target_ridge: bool = False,  # WARNING: leaks y into latents across folds
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return per-component features (n_components, K) and component dict.

    K = n_latent_svd (U factor) + n_latent_nmf + 2 (ridge coef per target) +
        1 (component frequency).
    Returns:
        comp_feats: (n_components, K) matrix of per-component latent features
        train_scenario_feats: (n_train_scenarios, K) = M @ comp_feats
        comp_to_idx: mapping from component name to index
    """
    all_comps, c2i = build_component_ids(mix_train, mix_test)
    M_train, train_sids = build_mass_matrix(mix_train, c2i)
    M_test, test_sids = build_mass_matrix(mix_test, c2i)

    n_comp = len(all_comps)

    # 1) SVD of concatenated mass matrix (train+test) — captures co-occurrence
    M_all = np.vstack([M_train, M_test])
    svd = TruncatedSVD(n_components=n_latent_svd, random_state=0)
    U_all = svd.fit_transform(M_all)
    Vt = svd.components_  # (n_latent, n_components)

    # Per-component SVD features = columns of Vt scaled by singular values
    svd_feats = (Vt * np.sqrt(svd.singular_values_).reshape(-1, 1)).T  # (n_components, n_latent)

    # 2) NMF on train mass (non-negative)
    try:
        nmf = NMF(n_components=min(n_latent_nmf, n_comp-1), init='nndsvd',
                  random_state=0, max_iter=400)
        _ = nmf.fit_transform(M_all + 1e-9)
        nmf_feats = nmf.components_.T  # (n_components, n_latent_nmf)
    except Exception:
        nmf_feats = np.zeros((n_comp, n_latent_nmf), dtype=np.float32)

    # 3) Ridge on train: y ~ M @ θ per target (linear effect of each component).
    # This LEAKS y across CV folds → disabled by default. Keep SVD+NMF which only
    # use the mass matrix (no targets).
    if use_target_ridge:
        ridge_feats = np.zeros((n_comp, 2), dtype=np.float32)
        for t in range(2):
            r = Ridge(alpha=ridge_alpha, fit_intercept=False)
            y_t = np.arcsinh(y_train[:, t] / (10.0 if t == 0 else 20.0))
            r.fit(M_train, y_t)
            ridge_feats[:, t] = r.coef_
    else:
        ridge_feats = np.zeros((n_comp, 0), dtype=np.float32)

    # 4) Frequency features
    freq = (M_all > 0).sum(axis=0).reshape(-1, 1).astype(np.float32)  # how many scenarios use each component

    comp_feats = np.hstack([svd_feats, nmf_feats, ridge_feats, freq]).astype(np.float32)
    # Standardize columns (robust)
    mu = comp_feats.mean(axis=0, keepdims=True)
    sd = comp_feats.std(axis=0, keepdims=True) + 1e-6
    comp_feats = (comp_feats - mu) / sd
    comp_feats = np.clip(comp_feats, -5, 5)

    return comp_feats, c2i, {"n_latent_svd": n_latent_svd, "n_latent_nmf": n_latent_nmf,
                             "sids_train": train_sids, "sids_test": test_sids,
                             "M_train": M_train, "M_test": M_test}
