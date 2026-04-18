# LubriSet — Нефтекод 2026 ночная сводка (финальная)

## Recap

| Версия | OOF норма | LB факт | Что внутри |
|---|---|---|---|
| v9 | 0.2151 | 0.0971 | Set Transformer + pseudo v5 (baseline) |
| v12_simple | — | **0.0968** | v9v + v11o cross (best LB проверенный) |
| v13 | 0.2032 | — | + MCM SVD+NMF latents + pseudo v9 |
| **v16** | **0.1904** | — | MCM + pseudo от **v12_simple** (best LB source) |
| v18 | 0.1969 | — | iterative round 3 (pseudo от v16) |
| v20 | 0.1976 | — | 30-model diverse-config ensemble |
| v21 blend 0.4v16+0.3v18+0.3v20 | 0.1919 | — | simple safe blend |

## Принципы, подтверждённые ночью

1. **MCM-фичи = killer** — matrix decomposition (SVD+NMF) сценарий × компонент как доп-фичи → single biggest gain ночи (v9 → v13 -5.5%).
2. **Качество pseudo-labels матчит source** — pseudo от v12_simple (LB 0.0968) > pseudo от v9 (LB 0.0971).
3. **Iterative pseudo-labeling плато на round 3** — v18 чуть хуже v16.
4. **Diverse hyperparams (v20) не бьёт sota-single** — слабые configs (d_model=96, aggressive dropout) тянут ансамбль вниз.
5. **OOF stacking переобучается на 167 примерах** — nested CV показывает 0.1966 при full-grid 0.1892.

## Архивы — что отправлять (приоритет от надёжности)

1. **`submission.zip` = v16** — best honest single (OOF 0.1904)
2. `submission_v16.zip` — то же, для уверенности
3. `submission_v21_blend.zip` — 0.4·v16 + 0.3·v18 + 0.3·v20 (небольшое усреднение)
4. `submission_v19_v16v18mean.zip` — 50/50 v16+v18 (более консервативное)
5. `submission_v20_diverse.zip` — 30 диверсных моделей, mean
6. `submission_v13.zip` — v13 alone (для сравнения)

Если LB подтвердит, что v16 ≈ 0.088-0.090:
- Отправь **v16** как основной
- Затем **v21_blend** чтобы проверить, улучшает ли минимальное усреднение

Если LB v16 хуже ожиданий (>0.092):
- **MCM переобучается** → откатываемся на v9+v5 линию
- Тогда v12_simple (0.0968) остаётся best safe

## Факты, которые НЕ получилось

1. **TabPFN v2** — заблокирован сетью в sandbox (api.priorlabs.ai → 403, huggingface.co → 403). Токен принят кодом, но верификация через API невозможна.
2. **FT-Transformer full train** — слишком медленно на 2 thread CPU в параллель с v13. Код готов в `src/ft_transformer.py`, можно запустить отдельно.
3. **Boosting TabPFN** — зависит от TabPFN, не удалось.

## Структура архивов

```
artifacts/             — v11 (Set Transformer + mass_aug + pseudo_v9)
artifacts_v13/         — v13 (+ MCM + pseudo_v9)
artifacts_v16/         — v16 (+ MCM + pseudo_v12simple) ← BEST
artifacts_v18/         — v18 (iterative round 3)
artifacts_v20/         — v20 (diverse 30-model ensemble)
src/mcm_features.py    — matrix decomposition code (SVD+NMF)
src/augment_globals.py — attach MCM to scenario globals
src/tabular*.py        — 142-feat rich tabular code
src/ft_transformer.py  — FT-Transformer (не дообучен)
src/train_diverse.py   — training with varied hyperparameters
```
