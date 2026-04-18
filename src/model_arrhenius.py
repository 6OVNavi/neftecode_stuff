"""LubriSet with Arrhenius-parameterized oxidation head.

The viscosity head is unchanged (MLP → scalar in asinh space).  The oxidation
head instead outputs three physical quantities per scenario:
  log A        (pre-exponential factor, in natural log)
  Ea_over_R    (activation energy divided by the gas constant, units of K)
  AO_logit     (antioxidant depletion fraction before sigmoid)

We read T (°C) and t (h) from the condition vector, convert T to Kelvin, then
compute
  EOT_raw = exp(log A - Ea/R · 1/T_K) · t · (1 - AO)
and transform with asinh to match the loss space used everywhere else.

Parameter ranges are bounded to physically plausible values for lubricant
oxidation (Ea ~ 60-170 kJ/mol, A ~ 10^8-10^13 1/h):
  log A ∈ [18, 32]        via  26 + 7·tanh(raw)
  Ea/R  ∈ [7000, 21000]   via  14000 + 7000·tanh(raw)
  AO    ∈ [0, 0.95]       via  0.95·sigmoid(raw)

With centers picked so that at T=433 K (160 °C) and t=168 h the default
prediction is near the dominant-regime mean of the training set (EOT ≈ 30).
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MAB, SAB, PMA, ComponentEncoder, TypePairwise


R_KELVIN = 1.0  # we express Ea as Ea/R directly in Kelvin; physical R is implicit


class LubriSetArrhenius(nn.Module):
    """Same trunk as LubriSet but the oxidation head is physics-shaped."""

    def __init__(
        self,
        n_components: int,
        n_types: int,
        n_props: int,
        condition_dim: int,
        global_dim: int = 0,
        pair_channels: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        n_targets: int = 2,        # must be 2 to keep API
        dropout: float = 0.1,
        id_dropout: float = 0.25,
        asinh_scale_ox: float = 20.0,
    ):
        super().__init__()
        assert n_targets == 2, "Arrhenius variant hardcoded for 2 targets"
        self.asinh_scale_ox = asinh_scale_ox

        self.encoder = ComponentEncoder(
            n_components, n_types, n_props, d_model, id_dropout=id_dropout
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(condition_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        self.global_dim = global_dim
        if global_dim > 0:
            self.global_mlp = nn.Sequential(
                nn.Linear(global_dim, d_model), nn.GELU(),
                nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
            )
        self.layers = nn.ModuleList(
            [SAB(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        # Two PMA seeds: one for viscosity, one for the Arrhenius triple.
        self.pma = PMA(d_model, n_heads, k=2, dropout=dropout)
        self.pair_channels = pair_channels
        self.type_pairwise = TypePairwise(n_types=n_types, n_channels=pair_channels)
        head_in = d_model + global_dim + pair_channels
        # Viscosity head = standard scalar head
        self.head_visc = nn.Sequential(
            nn.Linear(head_in, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        # Arrhenius head outputs 3 parameters: logA residual, Ea/R positive, AO logit
        self.head_arr = nn.Sequential(
            nn.Linear(head_in, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 3),  # log A raw, Ea raw, AO raw
        )
        # Learnable offset so the default forward lands near the data mean.
        # Train data EOT mean ≈ 50, asinh(50/20) ≈ 1.7, so we want log_eot ≈ log(50)=3.9.
        self.log_eot_offset = nn.Parameter(torch.tensor(3.9))

    def _arrhenius_eot(self, arr_params: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """arr_params: (B, 3) raw outputs.  conditions[:, 0]=T in C, [:, 1]=t in h.
        Returns EOT in raw units (A/cm).

        Parameterisation:
          - logA is unconstrained (the exponential gives positive A).  The
            model's freedom to pick large logA is balanced by Ea/R via the
            Arrhenius law.
          - Ea/R is positive via softplus, shifted so zero raw ⇒ ~11000 K
            (activation energy ~90 kJ/mol, typical for oxidation of mineral
            oils).
          - AO depletion is in (0, 0.95).
          - log_eot_offset is a single learnable scalar that absorbs the
            mean, making the default forward land near data mean without
            forcing the NN outputs to do both (stability).
        """
        logA_resid, Ea_raw, AO_raw = arr_params.unbind(dim=-1)
        Ea_over_R = F.softplus(Ea_raw) * 3000.0 + 4000.0   # ≥ 4000 K, default ~11000
        AO = 0.95 * torch.sigmoid(AO_raw)                   # [0, 0.95]

        T_C = conditions[:, 0]
        t_h = conditions[:, 1]
        T_K = T_C + 273.15
        log_eot = (self.log_eot_offset
                   + logA_resid                            # unbounded residual
                   - Ea_over_R * (1.0 / T_K - 1.0 / 433.15)  # centered at 160 °C
                   + torch.log(torch.clamp(t_h / 168.0, min=0.1))
                   + torch.log(torch.clamp(1.0 - AO, min=1e-3)))
        eot_raw = torch.exp(log_eot)
        return eot_raw

    def forward(self, comp_ids, type_ids, props, miss, mass, is_new,
                conditions, pad_mask, global_feats=None):
        """Returns (B, 2) predictions in the ASINH target space to match the
        existing loss function."""
        comp_tokens = self.encoder(comp_ids, type_ids, props, miss, mass, is_new)
        cond_token = self.cond_mlp(conditions).unsqueeze(1)
        comp_tokens = comp_tokens * (1.0 + mass.unsqueeze(-1))
        tokens = [cond_token, comp_tokens]
        prefix_tokens = 1
        if self.global_dim > 0 and global_feats is not None:
            gtok = self.global_mlp(global_feats).unsqueeze(1)
            tokens.insert(0, gtok)
            prefix_tokens = 2
        x = torch.cat(tokens, dim=1)
        B = pad_mask.size(0)
        prefix_pad = torch.zeros(B, prefix_tokens, dtype=torch.bool, device=pad_mask.device)
        full_mask = torch.cat([prefix_pad, pad_mask], dim=1)
        for layer in self.layers:
            x = layer(x, full_mask)
        pooled = self.pma(x, full_mask)  # (B, 2, D)

        pair_feat = self.type_pairwise(type_ids, mass, pad_mask)
        extras = [pair_feat]
        if self.global_dim > 0 and global_feats is not None:
            extras.insert(0, global_feats)
        extra_cat = torch.cat(extras, dim=-1)

        pooled_full = torch.cat(
            [pooled, extra_cat.unsqueeze(1).expand(-1, 2, -1)], dim=-1
        )
        visc_asinh = self.head_visc(pooled_full[:, 0]).squeeze(-1)
        arr_params = self.head_arr(pooled_full[:, 1])  # (B, 3)

        eot_raw = self._arrhenius_eot(arr_params, conditions)
        eot_asinh = torch.asinh(eot_raw / self.asinh_scale_ox)

        return torch.stack([visc_asinh, eot_asinh], dim=-1)
