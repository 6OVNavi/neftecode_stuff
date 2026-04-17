# LubriSet — Нефтекод 2026

Set Transformer ансамбль для предсказания результатов DOT-теста моторных масел.

## Структура

```
src/
  data.py     # Загрузка, свойства, построение scenario-сэмплов
  model.py    # Set Transformer (MAB/SAB/PMA) + кодер компонентов
  train.py    # 5-fold scenario CV × N сидов, deep ensemble
  infer.py    # Генерация predictions.csv
inference.ipynb   # Ноутбук для инференса (требование хакатона)
artifacts/        # Веса обученных моделей + OOF
```

## Запуск

```bash
pip install -r requirements.txt

# Обучение (5 folds × 2 seeds × 250 epochs = 10 моделей)
python -m src.train --n_folds 5 --n_seeds 2 --epochs 250 --out_dir artifacts

# Инференс
python -m src.infer --data_dir . --artifacts artifacts --out predictions.csv
```

Или через Docker:

```bash
docker build -t lubriset .
docker run --rm -v $(pwd):/workspace lubriset
```

## Ключевые идеи

1. **Set Transformer (Lee 2019)** с ISAB/PMA для переменного числа компонентов (6-20).
2. **Cross-attention с токеном условий** — температура, время, биотопливо %, катализатор как 14-D one-hot+continuous вектор, проецируется в токен и участвует в self-attention.
3. **Component encoding** объединяет:
   - Learnable ID embedding (с id-dropout для робастности к новым компонентам).
   - Type embedding (9 классов: base oil, ZDDP, AO, ...).
   - MLP на физ-хим свойствах (top-30 по покрытию) + missing mask.
4. **Mass-weighted token scaling** — токены умножаются на (1 + mass_fraction), чтобы мажорные компоненты доминировали.
5. **Sign-preserving log1p трансформация таргета вязкости** — работает с правоскошенностью + отрицательными значениями.
6. **Component dropout** — случайное удаление компонента (p=0.15) при обучении → forces robustness к 8 новым компонентам в test.
7. **Deep ensemble** через K-fold × seed.
8. **Обработка аномалий**:
   - Дубли (component, scenario) — агрегируются суммой масс (91 случай).
   - Массы нормализуются к сумме 1 внутри сценария.
   - Плотности приводятся к единой единице.
   - «Слепые» компоненты (без свойств) — 0 + missing mask.
