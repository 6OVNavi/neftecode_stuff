"""Set Transformer for oil-mixture property prediction.

Inputs per scenario:
  - component tokens (n components, variable)
    each built from: [type_embed + comp_id_embed + MLP(props||miss_mask||mass)]
  - condition token (from conditions vector)
Stack = [cond_token, comp_1, ..., comp_n]; self-attention over them.
Pool via PMA with 2 seeds (one per target).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    """Multihead Attention Block (Lee et al., 2019)."""

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln0 = nn.LayerNorm(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, q, k, mask_k=None):
        # mask_k: bool, True = pad (to ignore).
        h, _ = self.mha(q, k, k, key_padding_mask=mask_k, need_weights=False)
        x = self.ln0(q + h)
        return self.ln1(x + self.ff(x))


class SAB(nn.Module):
    """Self-attention block with padding mask."""

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mab = MAB(dim, n_heads, dropout)

    def forward(self, x, mask):
        return self.mab(x, x, mask)


class PMA(nn.Module):
    """Pooling by multihead attention with k learnable seeds."""

    def __init__(self, dim: int, n_heads: int, k: int, dropout: float = 0.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, k, dim) * 0.02)
        self.mab = MAB(dim, n_heads, dropout)

    def forward(self, x, mask):
        b = x.size(0)
        q = self.seeds.expand(b, -1, -1)
        return self.mab(q, x, mask)


class ComponentEncoder(nn.Module):
    def __init__(
        self,
        n_components: int,
        n_types: int,
        n_props: int,
        d_model: int = 128,
        id_dropout: float = 0.25,
    ):
        super().__init__()
        self.comp_emb = nn.Embedding(n_components, d_model)
        self.type_emb = nn.Embedding(n_types, d_model)
        # Props + miss_mask + log(mass+eps) + mass.
        prop_in = 2 * n_props + 2
        self.prop_mlp = nn.Sequential(
            nn.Linear(prop_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.ln = nn.LayerNorm(d_model)
        self.id_dropout = id_dropout
        nn.init.normal_(self.comp_emb.weight, std=0.02)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        # <UNK> is index 0 for both.
        with torch.no_grad():
            self.comp_emb.weight[0].zero_()

    def forward(self, comp_ids, type_ids, props, miss, mass, is_new):
        # comp_ids: (B, N), props: (B, N, P), miss: (B, N, P), mass: (B, N), is_new: (B, N)
        # During training, randomly drop comp_id embedding to force reliance on props.
        if self.training and self.id_dropout > 0:
            drop = (torch.rand_like(mass) < self.id_dropout).float()
        else:
            drop = torch.zeros_like(mass)
        # Always zero out component embedding for "new in test" components to avoid leakage
        # of an <UNK>-like single embedding; we add nothing for them.
        drop = torch.clamp(drop + is_new, 0.0, 1.0)
        comp_vec = self.comp_emb(comp_ids) * (1.0 - drop).unsqueeze(-1)
        type_vec = self.type_emb(type_ids)
        mass_feat = torch.stack([mass, torch.log1p(mass * 100.0)], dim=-1)
        prop_in = torch.cat([props, miss, mass_feat], dim=-1)
        prop_vec = self.prop_mlp(prop_in)
        return self.ln(comp_vec + type_vec + prop_vec)


class LubriSet(nn.Module):
    def __init__(
        self,
        n_components: int,
        n_types: int,
        n_props: int,
        condition_dim: int,
        global_dim: int = 0,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        n_targets: int = 2,
        dropout: float = 0.1,
        id_dropout: float = 0.25,
    ):
        super().__init__()
        self.encoder = ComponentEncoder(
            n_components, n_types, n_props, d_model, id_dropout=id_dropout
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(condition_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.global_dim = global_dim
        if global_dim > 0:
            self.global_mlp = nn.Sequential(
                nn.Linear(global_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
        self.layers = nn.ModuleList(
            [SAB(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        # One seed per target.
        self.pma = PMA(d_model, n_heads, k=n_targets, dropout=dropout)
        # Head receives pooled vector + raw global features concatenated.
        head_in = d_model + global_dim
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_in, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            for _ in range(n_targets)
        ])
        self.n_targets = n_targets

    def forward(self, comp_ids, type_ids, props, miss, mass, is_new,
                conditions, pad_mask, global_feats=None):
        """Returns (B, n_targets) predictions in the transformed space."""
        comp_tokens = self.encoder(comp_ids, type_ids, props, miss, mass, is_new)
        cond_token = self.cond_mlp(conditions).unsqueeze(1)  # (B,1,D)
        # Scale by mass fraction: emphasizes major components.
        comp_tokens = comp_tokens * (1.0 + mass.unsqueeze(-1))
        tokens = [cond_token, comp_tokens]
        prefix_tokens = 1  # cond token
        if self.global_dim > 0 and global_feats is not None:
            gtok = self.global_mlp(global_feats).unsqueeze(1)
            tokens.insert(0, gtok)
            prefix_tokens = 2
        x = torch.cat(tokens, dim=1)
        # Extend pad_mask to cover prepended tokens (never padded).
        B = pad_mask.size(0)
        prefix_pad = torch.zeros(B, prefix_tokens, dtype=torch.bool, device=pad_mask.device)
        full_mask = torch.cat([prefix_pad, pad_mask], dim=1)
        for layer in self.layers:
            x = layer(x, full_mask)
        pooled = self.pma(x, full_mask)  # (B, n_targets, D)
        if self.global_dim > 0 and global_feats is not None:
            pooled = torch.cat([pooled, global_feats.unsqueeze(1).expand(-1, self.n_targets, -1)], dim=-1)
        outs = [self.heads[i](pooled[:, i]).squeeze(-1) for i in range(self.n_targets)]
        return torch.stack(outs, dim=-1)
