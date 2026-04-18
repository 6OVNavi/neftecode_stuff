# LubriSet — Нефтекод 2026 ночная сводка

## Текущий best OOF: **0.1904** (v16), линейный прогноз LB ~0.0858

## Что сделано

| Версия | OOF норма | LB факт | Что внутри |
|---|---|---|---|
| v2 | 0.232 | 0.0992 | Set Transformer + 61 prop + asinh |
| v5 (v2v+v4o) | 0.226 | 0.0978 | target-specialized |
| v9 | 0.215 | 0.0971 | v2 + pseudo v5 (1 round) |
| v12_simple | — | **0.0968** | v9v + v11o cross |
| v13 | 0.203 | — | + MCM latents + pseudo v9 |
| v16 | **0.1904** | — | MCM + pseudo v12_simple |
| v18 | 0.197 | — | iterative round 3 (pseudo v16) |
| v19 (v16+v18) | 0.191 | — | safe blend |
| v19_4blend | 0.189 (OOF) | — | агрессивный OOF-блeнд |

## Главный вывод

**v16 — ПИК одиночной модели. Blend не улучшает существенно** (nested CV показывает что блендинг ≈ v16 alone из-за overfit'а OOF-весов на 167 примерах).

## Что отправлять (в порядке приоритета)

1. **`submission.zip`** (= v16) — безопасный, лучший OOF на одной модели
2. `submission_v19_v16v18mean.zip` — safe blend v16+v18, tiny diff
3. `submission_v13.zip` — проверить вклад MCM vs v9
4. `submission_v19_4blend.zip` — risk-on: OOF-оптимальный 4-блeнд

## Что еще в процессе

- **v20** (`artifacts_v20/`): diverse 30-model ensemble (5 folds × 6 различных configs). Если дообучится — будет `submission_v20.zip`.
- TabPFN: **не заработал** — proxy блокирует api.priorlabs.ai и HF.

## Ключевые находки ночи

1. **MCM-latents** (матричная декомпозиция scenario × component → SVD+NMF фичи сценария) дали прорыв: v9 → v13 **−5.5% OOF** одной фичей.
2. **Pseudo-labels от BEST-LB модели** (v12_simple) > pseudo-labels от v9. Важно, чей сигнал ретранслируешь.
3. **Iterative pseudo-labels** работают 1 раз (v9→v13→v16), но 3-й round (v16→v18) дал диминишинг.
4. **Blend overfit**: грид-сёрч OOF-весов показывает 0.189, nested CV показывает 0.197. Реалия честно ≈ v16 alone.
5. **TabPFN v2 требует выход в сеть** — в sandbox заблокирован (api.priorlabs.ai: 403, huggingface: 403).

## Куда дальше если LB подтвердит

Если v16 даст LB ~0.088-0.090 (top-10):
- ensemble с v13 + v18 для страховки  
- 2 round pseudo-label cycle с последним LB лидером
- v20 diverse ensemble (в процессе)

Если v16 даст LB хуже ожиданий (>0.093):
- overfit на MCM → откатиться на v9+v5 линию
- попробовать чистый FT-Transformer от scratch
- decrypt-API аугментация от организаторов (добавить literature data в train)
