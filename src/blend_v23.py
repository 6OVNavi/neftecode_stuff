"""v23: Caruana-style post-hoc weighted blend + fold-wise isotonic calibration.

Reads OOF + test predictions from existing v16/v18/v20 runs (and v22 once ready),
learns a convex blend that minimises the competition normalised-MAE score on OOF,
then applies isotonic calibration per target to the blended OOF predictions, and
uses the learnt weights + calibrators to produce final test predictions.

Inspired by:
- TabArena 2506.16791: post-hoc weighted ensemble of tuned HP configs is the
  single most underrated trick; promotes NN methods above CatBoost on average.
- Open Polymer Challenge post-competition report 2512.08896: 3rd place gained
  5.5% from fold-wise linear/isotonic calibration of OOF predictions.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression


TEST_COLS = (
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
    "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm",
)


def score(pred, true, std_v, std_o):
    mv = float(np.mean(np.abs(pred[:, 0] - true[:, 0])))
    mo = float(np.mean(np.abs(pred[:, 1] - true[:, 1])))
    return mv / std_v / 2.0 + mo / std_o / 2.0, mv, mo


def load_run(run_dir: Path):
    oof = pd.read_csv(run_dir / "oof_predictions.csv")
    test = pd.read_csv(run_dir / "predictions_raw_headers.csv")
    return oof, test


def convex_blend(oofs, truth, std_v, std_o):
    """Find non-negative weights on the simplex minimising the OOF score."""
    n_runs = len(oofs)
    stack = np.stack(oofs, axis=0)  # (n_runs, N, 2)

    def loss(w):
        w = np.clip(w, 0, None)
        w = w / (w.sum() + 1e-12)
        pred = (w[:, None, None] * stack).sum(axis=0)
        s, _, _ = score(pred, truth, std_v, std_o)
        return s

    best = None
    # multi-start to avoid local minima
    for seed in range(20):
        rng = np.random.default_rng(seed)
        x0 = rng.dirichlet(np.ones(n_runs))
        res = minimize(loss, x0, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
        if best is None or res.fun < best.fun:
            best = res
    w = np.clip(best.x, 0, None)
    w = w / (w.sum() + 1e-12)
    return w, float(best.fun)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["artifacts_v16", "artifacts_v18", "artifacts_v20"],
                    help="Artifact dirs to blend, order matters (for logging).")
    ap.add_argument("--out_dir", default="artifacts_v23")
    ap.add_argument("--calibrate", action="store_true",
                    help="Apply isotonic calibration on blended OOF per target.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)

    # --- load all runs -------------------------------------------------------
    oofs_df, tests_df = [], []
    for r in args.runs:
        oof, test = load_run(Path(r))
        oofs_df.append(oof.sort_values("scenario_id").reset_index(drop=True))
        tests_df.append(test.sort_values("scenario_id").reset_index(drop=True))

    # truth (same across runs, take from first)
    base_oof = oofs_df[0]
    truth = base_oof[["true_viscosity", "true_oxidation"]].values.astype(np.float32)
    std_v = float(base_oof["true_viscosity"].std())
    std_o = float(base_oof["true_oxidation"].std())

    oof_stack = [df[["oof_viscosity", "oof_oxidation"]].values.astype(np.float32)
                 for df in oofs_df]

    # test predictions
    test_ids = tests_df[0]["scenario_id"].values
    test_stack = [df[list(TEST_COLS)].values.astype(np.float32) for df in tests_df]

    # --- per-run baselines ---------------------------------------------------
    print("\n=== per-run OOF scores ===")
    for name, oof in zip(args.runs, oof_stack):
        s, mv, mo = score(oof, truth, std_v, std_o)
        print(f"  {name}: norm={s:.4f}  mv={mv:.3f}  mo={mo:.3f}")

    # --- Caruana-style weighted blend ---------------------------------------
    weights, blend_score = convex_blend(oof_stack, truth, std_v, std_o)
    print(f"\n=== Caruana blend ===")
    for r, w in zip(args.runs, weights):
        print(f"  w[{r}] = {w:.3f}")
    print(f"  blended OOF norm = {blend_score:.4f}")

    blended_oof = sum(w * o for w, o in zip(weights, oof_stack))
    blended_test = sum(w * t for w, t in zip(weights, test_stack))

    # --- isotonic calibration (optional) ------------------------------------
    if args.calibrate:
        # Honest check: leave-one-out isotonic.  Fitting isotonic on all 167
        # OOF rows and scoring on the same 167 lets isotonic memorise; LOO
        # removes that leak and tells us the real generalisation gain.
        print("\n=== LOO isotonic calibration per target (honest estimate) ===")
        N = blended_oof.shape[0]
        loo_oof = np.empty_like(blended_oof)
        for j in range(2):
            for i in range(N):
                mask = np.ones(N, dtype=bool); mask[i] = False
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(blended_oof[mask, j], truth[mask, j])
                loo_oof[i, j] = iso.transform(blended_oof[i:i+1, j])[0]
        s, mv, mo = score(loo_oof, truth, std_v, std_o)
        print(f"  LOO isotonic OOF: norm={s:.4f}  mv={mv:.3f}  mo={mo:.3f}")

        # Fit final calibrators on all OOF and apply to test.
        cal_test = np.empty_like(blended_test)
        for j in range(2):
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(blended_oof[:, j], truth[:, j])
            cal_test[:, j] = iso.transform(blended_test[:, j])
        final_oof, final_test = loo_oof, cal_test
    else:
        final_oof, final_test = blended_oof, blended_test

    # --- save ----------------------------------------------------------------
    pd.DataFrame({
        "scenario_id": base_oof["scenario_id"].values,
        "oof_viscosity": final_oof[:, 0], "oof_oxidation": final_oof[:, 1],
        "true_viscosity": truth[:, 0], "true_oxidation": truth[:, 1],
    }).to_csv(out_dir / "oof_predictions.csv", index=False)

    pd.DataFrame({
        "scenario_id": test_ids,
        TEST_COLS[0]: final_test[:, 0],
        TEST_COLS[1]: final_test[:, 1],
    }).to_csv(out_dir / "predictions_raw_headers.csv", index=False)

    with open(out_dir / "blend_weights.json", "w") as f:
        json.dump({
            "runs": list(args.runs),
            "weights": [float(w) for w in weights],
            "oof_blend_score": blend_score,
            "calibrate": bool(args.calibrate),
        }, f, indent=2)
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
