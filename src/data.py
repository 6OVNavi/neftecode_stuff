"""Data loading and feature engineering for Nefticode 2026 / Daimler Oxidation Test.

Produces, per scenario:
  - set of component feature vectors (variable length)
  - scalar condition vector
  - target vector (train only)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Raw column names (some contain pipes/commas).
COL_SCENARIO = "scenario_id"
COL_COMP = "Компонент"
COL_BATCH = "Наименование партии"
COL_MASS = "Массовая доля, %"
COL_TEMP = "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C"
COL_TIME = "Время испытания | - Daimler Oxidation Test (DOT), ч"
COL_BIOFUEL = "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс"
COL_CAT = "Дозировка катализатора, категория"
COL_TARGET_VISC = "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %"
COL_TARGET_OX = "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"

COMPONENT_TYPES = [
    "Базовое_масло",
    "Загуститель",
    "Антиоксидант",
    "Детергент",
    "Дисперсант",
    "Противоизносная_присадка",
    "Соединение_молибдена",
    "Депрессорная_присадка",
    "Антипенная_присадка",
]

# Top numeric properties by coverage (picked via EDA, in order of usefulness).
TOP_PROPERTIES = [
    "Кинематическая вязкость, при 100°C, ASTM D445",
    "Кинематическая вязкость, при 40°C, ASTM D445",
    "Массовая доля фосфора, ASTM D6481",
    "Массовая доля кальция, ASTM D6481",
    "Массовая доля цинка, ASTM D6481",
    "Массовая доля серы, ASTM D6481",
    "Индекс вязкости, ГОСТ 25371",
    "Динамическая вязкость CCS -30°C, ASTM D5293",
    "Температура застывания, ГОСТ 20287, метод Б",
    "Динамическая вязкость CCS -20°C, ASTM D5293",
    "Динамическая вязкость CCS -25°C, ASTM D5293",
    "Щелочное число, ASTM D2896",
    "Динамическая вязкость CCS -15°C, ASTM D5293",
    "Испаряемость по NOACK, ASTM D5800",
    "Последовательность 1 | ASTM D892",
    "Последовательность 3 | ASTM D892",
    "Последовательность 2 | ASTM D892",
    "Деэм.время | ASTM D1401",
    "Деэм.масло | ASTM D1401",
    "Деэм.эмульсия | ASTM D1401",
    "Деэм.вода | ASTM D1401",
    "Динамическая вязкость CCS -35°C, ASTM D5293",
    "Отношение Мыло/Основание",
    "Содержание MgCO3, CaCO3, % масс.",
    "Содержание мыла, % масс.",
    "Содержание масла, % масс.",
    "Содержание металла (Ca/Mg), % масс.",
    "Щелочное число, ГОСТ 11362",
    "Группа по API",
    "Атомное отношение P:Zn",
]


def _to_float(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    # Parse things like "<10", ">95" as soft bounds.
    if s.startswith("<"):
        try:
            return float(s[1:].replace(",", ".")) * 0.5
        except Exception:
            return np.nan
    if s.startswith(">"):
        try:
            return float(s[1:].replace(",", ".")) * 1.1
        except Exception:
            return np.nan
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def component_type(comp: str) -> str:
    m = re.match(r"(.+)_\d+$", comp)
    return m.group(1) if m else comp


def normalize_units(pr: pd.DataFrame) -> pd.DataFrame:
    pr = pr.copy()
    # Density unit fix: if kg/m3 (>10), divide by 1000.
    mask_dens = pr["property"].str.contains("Плотность", na=False) & (pr["value_num"] > 10)
    pr.loc[mask_dens, "value_num"] = pr.loc[mask_dens, "value_num"] / 1000.0
    return pr


def load_properties(path: str) -> pd.DataFrame:
    pr = pd.read_csv(path)
    pr.columns = ["component", "batch", "property", "unit", "value"]
    pr["value_num"] = pr["value"].apply(_to_float)
    pr = normalize_units(pr)
    return pr


def build_component_property_table(pr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Return wide property tables per (component, batch) and per component (typical).

    Returns:
        wide_batch: index=(component, batch), columns=TOP_PROPERTIES
        wide_typ  : index=component, columns=TOP_PROPERTIES (with typical fallback + batch mean)
        mu, sd    : scaler params per property
    """
    pr = pr[pr["property"].isin(TOP_PROPERTIES)].copy()

    wide_batch = pr.pivot_table(
        index=["component", "batch"],
        columns="property",
        values="value_num",
        aggfunc="mean",
    ).reindex(columns=TOP_PROPERTIES)

    # Per-component table: prefer mean of non-typical batches; fallback to typical.
    non_typ = pr[pr["batch"].astype(str) != "typical"]
    typ = pr[pr["batch"].astype(str) == "typical"]
    wide_non_typ = non_typ.pivot_table(
        index="component", columns="property", values="value_num", aggfunc="mean"
    ).reindex(columns=TOP_PROPERTIES)
    wide_typ = typ.pivot_table(
        index="component", columns="property", values="value_num", aggfunc="mean"
    ).reindex(columns=TOP_PROPERTIES)
    wide_comp = wide_non_typ.combine_first(wide_typ)

    # Standardize (robust via median/MAD) ignoring NaNs.
    vals = wide_comp.values.astype(float)
    mu = np.nanmedian(vals, axis=0)
    sd = 1.4826 * np.nanmedian(np.abs(vals - mu), axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return wide_batch, wide_comp, mu, sd


@dataclass
class ScenarioSample:
    scenario_id: str
    comp_ids: np.ndarray       # (n,) int indices into component vocab
    type_ids: np.ndarray       # (n,) int indices into type vocab
    props: np.ndarray          # (n, P) standardized numeric props (0 = missing)
    miss_mask: np.ndarray      # (n, P) 1 if missing
    mass: np.ndarray           # (n,) raw mass fraction (normalized to sum 1 inside the scenario)
    conditions: np.ndarray     # (C,) condition vector
    is_new: np.ndarray         # (n,) 1 if component not in train vocab
    targets: np.ndarray | None # (2,) target_viscosity, target_oxidation (train only)


def build_condition_vector(temp: float, time: float, biofuel: float, cat: int) -> np.ndarray:
    # One-hot all conditions (all values known in both splits).
    temp_oh = [int(temp == 150), int(temp == 154), int(temp == 160)]
    time_oh = [int(time == 168), int(time == 216)]
    biofuel_oh = [int(biofuel == 0), int(biofuel == 5), int(biofuel == 7)]
    cat_oh = [int(cat == 1), int(cat == 2)]
    # Plus normalized continuous versions as backup.
    return np.array(temp_oh + time_oh + biofuel_oh + cat_oh +
                    [(temp - 155) / 5.0, (time - 192) / 24.0, biofuel / 7.0, float(cat - 1)],
                    dtype=np.float32)


CONDITION_DIM = 3 + 2 + 3 + 2 + 4  # 14


def build_scenario_samples(
    mix_df: pd.DataFrame,
    wide_batch: pd.DataFrame,
    wide_comp: pd.DataFrame,
    mu: np.ndarray,
    sd: np.ndarray,
    comp_vocab: dict[str, int],
    type_vocab: dict[str, int],
    train_comp_set: set[str] | None = None,
    is_train: bool = True,
) -> list[ScenarioSample]:
    samples: list[ScenarioSample] = []

    for scen_id, grp in mix_df.groupby(COL_SCENARIO, sort=False):
        # Aggregate duplicate (component, batch) rows: sum mass fractions (anomaly #1).
        grp = grp.copy()
        agg = (
            grp.groupby([COL_COMP, COL_BATCH], dropna=False)[COL_MASS]
            .sum()
            .reset_index()
        )
        n = len(agg)
        comp_ids = np.zeros(n, dtype=np.int64)
        type_ids = np.zeros(n, dtype=np.int64)
        props = np.zeros((n, len(TOP_PROPERTIES)), dtype=np.float32)
        miss = np.ones((n, len(TOP_PROPERTIES)), dtype=np.float32)
        mass = np.zeros(n, dtype=np.float32)
        is_new = np.zeros(n, dtype=np.float32)

        for i in range(len(agg)):
            comp = agg.iloc[i, 0]   # Компонент
            batch = agg.iloc[i, 1]  # Наименование партии
            m = agg.iloc[i, 2]      # Массовая доля, %
            ctype = component_type(comp)
            type_ids[i] = type_vocab.get(ctype, 0)
            comp_ids[i] = comp_vocab.get(comp, 0)  # 0 = <UNK>
            if train_comp_set is not None and comp not in train_comp_set:
                is_new[i] = 1.0
            mass[i] = float(m)

            # Batch-level props, then component-level fallback.
            key = (comp, batch)
            vals = None
            if key in wide_batch.index:
                vals = wide_batch.loc[key].values.astype(float)
            if vals is None or np.all(np.isnan(vals)):
                if comp in wide_comp.index:
                    vals = wide_comp.loc[comp].values.astype(float)
                else:
                    vals = np.full(len(TOP_PROPERTIES), np.nan, dtype=float)
            # Standardize; NaN -> 0 + mask=1.
            std = (vals - mu) / sd
            m_mask = np.isnan(std)
            std = np.where(m_mask, 0.0, std)
            # Clip extreme outliers.
            std = np.clip(std, -5.0, 5.0)
            props[i] = std.astype(np.float32)
            miss[i] = m_mask.astype(np.float32)

        # Normalize mass fractions to sum to 1 inside a scenario (anonymized transform).
        s = mass.sum()
        mass_norm = mass / (s + 1e-9)

        first = grp.iloc[0]
        cond = build_condition_vector(
            float(first[COL_TEMP]),
            float(first[COL_TIME]),
            float(first[COL_BIOFUEL]),
            int(first[COL_CAT]),
        )

        targets = None
        if is_train and COL_TARGET_VISC in grp.columns:
            y1 = float(first[COL_TARGET_VISC])
            y2 = float(first[COL_TARGET_OX])
            targets = np.array([y1, y2], dtype=np.float32)

        samples.append(
            ScenarioSample(
                scenario_id=str(scen_id),
                comp_ids=comp_ids,
                type_ids=type_ids,
                props=props,
                miss_mask=miss,
                mass=mass_norm.astype(np.float32),
                conditions=cond.astype(np.float32),
                is_new=is_new,
                targets=targets,
            )
        )
    return samples


def build_vocabs(mix_train: pd.DataFrame, mix_test: pd.DataFrame) -> tuple[dict, dict]:
    all_comps = sorted(set(mix_train[COL_COMP]).union(set(mix_test[COL_COMP])))
    # 0 is reserved for <UNK>.
    comp_vocab = {"<UNK>": 0}
    for c in all_comps:
        comp_vocab[c] = len(comp_vocab)
    type_vocab = {"<UNK>": 0}
    for t in COMPONENT_TYPES:
        type_vocab[t] = len(type_vocab)
    return comp_vocab, type_vocab


def target_transform(y: np.ndarray) -> np.ndarray:
    """Sign-preserving log1p for viscosity (skew), z-score later for oxidation.

    Expects shape (N, 2) where col 0 is viscosity (%), col 1 is EOT (A/cm).
    """
    y = y.astype(np.float32).copy()
    y[:, 0] = np.sign(y[:, 0]) * np.log1p(np.abs(y[:, 0]))
    return y


def target_inverse_transform(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32).copy()
    y[:, 0] = np.sign(y[:, 0]) * (np.expm1(np.abs(y[:, 0])))
    return y
