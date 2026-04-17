"""Build tabular (fixed-length) scenario features for TabPFN / FT-Transformer.

Strategy: each scenario → one row of ~80 features covering:
- Conditions (T, t, biofuel, cat) one-hot + numeric
- Type presence (mass sum per type, 9 dims)
- Physics aggregates (same as set-transformer globals: total P/Ca/Zn/S/TBN/N/B/water/Mo/NOACK, means VI/KV100/BDE/HOMO/LUMO/IP/API/steric, synergy flags)
- Per-type weighted property means for the most informative props per type
  (base oil: VI, KV100, NOACK, density; ZDDP: P, Zn, S; detergent: TBN, Ca, Mg;
  antioxidant: BDE, N, HOMO, LUMO; etc.)
- Count of unique component types used
- Heuristic "peroxide factor": biofuel * time / temperature_K
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import (
    CONDITION_DIM, COL_COMP, TOP_PROPERTIES, PHYS_AGG_SPEC, PHYS_AGG_NAMES,
    AGG_MODE, COMPONENT_TYPES,
    build_component_property_table, build_scenario_samples, build_vocabs,
    load_properties,
)

# For each type, we want weighted means of a subset of properties most
# relevant to its role. Using indices into TOP_PROPERTIES (see data.py).
_IDX = {name: i for i, name in enumerate(TOP_PROPERTIES)}

TYPE_SPECIFIC = {
    "Базовое_масло": [
        _IDX["Кинематическая вязкость, при 100°C, ASTM D445"],
        _IDX["Кинематическая вязкость, при 40°C, ASTM D445"],
        _IDX["Индекс вязкости, ГОСТ 25371"],
        _IDX["Испаряемость по NOACK, ASTM D5800"],
        _IDX["Плотность при 15°С, ASTM D4052"],
        _IDX["Плотность при 20°С, ASTM D4052"],
        _IDX["Температура застывания, ГОСТ 20287, метод Б"],
        _IDX["Группа по API"],
        _IDX["Содержание серы, % масс."],
        _IDX["Содержание насыщ. у/в"],
    ],
    "Антиоксидант": [
        _IDX["Энергия диссоциации связи Х-Н, ккал/моль"],
        _IDX["Содержание Азота"],
        _IDX["Энергия ВЗМО, эВ"],
        _IDX["Энергия НСМО, эВ"],
        _IDX["Потенциал ионизации,эВ"],
        _IDX["Активный Азот / Кислород, % масс. (N или O)"],
        _IDX["Стерический фактор, Å3"],
        _IDX["Масса гидрофобного хвоста, г/моль"],
    ],
    "Детергент": [
        _IDX["Щелочное число, ASTM D2896"],
        _IDX["Щелочное число, ГОСТ 11362"],
        _IDX["Массовая доля кальция, ASTM D6481"],
        _IDX["Массовая доля кальция | ASTM D6481"],
        _IDX["Содержание MgCO3, CaCO3, % масс."],
        _IDX["Отношение Мыло/Основание"],
    ],
    "Дисперсант": [
        _IDX["Содержание Азота"],
        _IDX["Активный Азот / Кислород, % масс. (N или O)"],
        _IDX["Содержание Бора"],
        _IDX["Кинематическая вязкость, при 100°C, ASTM D445"],
    ],
    "Противоизносная_присадка": [
        _IDX["Массовая доля фосфора, ASTM D6481"],
        _IDX["Массовая доля фосфора | ASTM D6481"],
        _IDX["Массовая доля цинка, ASTM D6481"],
        _IDX["Массовая доля цинка | ASTM D6481"],
        _IDX["Массовая доля серы, ASTM D6481"],
        _IDX["Массовая доля серы | ASTM D6481"],
        _IDX["Атомное отношение P:Zn"],
    ],
    "Соединение_молибдена": [
        _IDX["% масс. (Mo)"],
        _IDX["Содержание Азота"],
        _IDX["Кинематическая вязкость, при 100°C, ASTM D445"],
    ],
    "Загуститель": [
        _IDX["Кинематическая вязкость, при 100°C, ASTM D445"],
        _IDX["Индекс полидисперсности"],
        _IDX["Индекс вязкости, ГОСТ 25371"],
    ],
}

TYPE_FEATURE_COUNT = sum(len(v) for v in TYPE_SPECIFIC.values())


def build_tabular(samples) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    """Return X (n_samples, D), feature_names, y (n_samples, 2) or None."""
    rows = []
    ids = []
    ys = []
    feat_names = None
    for s in samples:
        feat = {}
        # Conditions (raw + one-hot).
        # cond vector is [T_oh3, t_oh2, bio_oh3, cat_oh2, T_norm, t_norm, bio_norm, cat-1]
        feat["T_150"] = float(s.conditions[0])
        feat["T_154"] = float(s.conditions[1])
        feat["T_160"] = float(s.conditions[2])
        feat["t_168"] = float(s.conditions[3])
        feat["t_216"] = float(s.conditions[4])
        feat["bio_0"] = float(s.conditions[5])
        feat["bio_5"] = float(s.conditions[6])
        feat["bio_7"] = float(s.conditions[7])
        feat["cat_1"] = float(s.conditions[8])
        feat["cat_2"] = float(s.conditions[9])
        # Numeric forms.
        T_raw = float(s.conditions[10]) * 5 + 155
        t_raw = float(s.conditions[11]) * 24 + 192
        bio_raw = float(s.conditions[12]) * 7
        feat["T_raw"] = T_raw
        feat["t_raw"] = t_raw
        feat["bio_raw"] = bio_raw
        # Interaction features (physics inspired).
        feat["bio_x_t"] = bio_raw * t_raw / 100.0
        feat["bio_over_T"] = bio_raw / max(T_raw - 140, 1)
        feat["peroxide_factor"] = bio_raw * t_raw / (T_raw + 273)

        # Global features (from globals vector).
        for i, name in enumerate(PHYS_AGG_NAMES):
            feat[f"agg_{name}"] = float(s.globals[i])
        # Type-presence floats.
        for i, t in enumerate(COMPONENT_TYPES):
            feat[f"type_mass_{t}"] = float(s.globals[len(PHYS_AGG_NAMES) + i])
        # Synergy flags.
        feat["syn_Mo_ZDDP"] = float(s.globals[-3])
        feat["syn_Mo_AO"] = float(s.globals[-2])
        feat["syn_ZDDP_AO"] = float(s.globals[-1])

        # Per-type weighted means of type-specific property subsets.
        # Build for each type: masses of that type, and their props (raw, un-standardized
        # approximation: using standardized props since we don't keep raw).
        # s.props is (n, P) STANDARDIZED; miss_mask 1 if missing; we need RAW.
        # We don't have raw here; use standardized + flag "missing" counts instead.
        # This is OK: TabPFN normalizes internally.
        n = len(s.comp_ids)
        for t, indices in TYPE_SPECIFIC.items():
            type_idx = -1
            for i, tt in enumerate(COMPONENT_TYPES):
                if tt == t:
                    type_idx = i
                    break
            mask = np.array([s.type_ids[k] == type_idx + 1 for k in range(n)], dtype=bool)  # +1 since 0=<UNK>
            # Need mapping from type_vocab; assume COMPONENT_TYPES order matches vocab minus UNK.
            for pi in indices:
                # Weighted mean of standardized prop value, falling back to 0 if all missing.
                vals = s.props[:, pi]
                mvals = s.miss_mask[:, pi]
                valid = (~mvals.astype(bool)) & mask
                if valid.any():
                    w = s.mass[valid]
                    v = vals[valid]
                    if w.sum() > 0:
                        feat[f"{t}_p{pi}"] = float((v * w).sum() / (w.sum() + 1e-9))
                    else:
                        feat[f"{t}_p{pi}"] = 0.0
                else:
                    feat[f"{t}_p{pi}"] = 0.0

        # Simple count features.
        feat["n_components"] = int(n)
        feat["mass_sum"] = float(s.mass.sum())
        # How many distinct types used.
        feat["n_types_used"] = int(len(set(s.type_ids.tolist())))

        # New-in-train flag (for test only).
        feat["n_new"] = float(s.is_new.sum())

        rows.append(feat)
        ids.append(s.scenario_id)
        if s.targets is not None:
            ys.append(s.targets)

    feat_names = list(rows[0].keys())
    X = np.array([[r[k] for k in feat_names] for r in rows], dtype=np.float32)
    y = np.array(ys, dtype=np.float32) if ys else None
    return X, feat_names, y, ids
