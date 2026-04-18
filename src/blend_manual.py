"""Manual-weight blends for v24.

Caruana OOF-optimal collapses to ~v16 because v16 has best OOF; but on the LB
v20 has best actual score (0.0964 vs v16 ~0.0977). OOF is not a reliable
proxy on this 167-row dataset. So we also produce weighted blends where the
weights come from LB performance (known externally) or equal-weight, and let
the leaderboard decide.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TEST_COLS = (
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
    "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm",
)


def score(pred, true, std_v, std_o):
    mv = float(np.mean(np.abs(pred[:, 0] - true[:, 0])))
    mo = float(np.mean(np.abs(pred[:, 1] - true[:, 1])))
    return mv / std_v / 2.0 + mo / std_o / 2.0, mv, mo


def load_run(run_dir: Path):
    oof = pd.read_csv(run_dir / "oof_predictions.csv").sort_values("scenario_id").reset_index(drop=True)
    test = pd.read_csv(run_dir / "predictions_raw_headers.csv").sort_values("scenario_id").reset_index(drop=True)
    return oof, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Artifact dirs in the order matching --weights.")
    ap.add_argument("--weights", nargs="+", type=float, required=True,
                    help="Non-negative weights; will be normalised to sum 1.")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    assert len(args.runs) == len(args.weights), "runs and weights must match in length"
    w = np.array(args.weights, dtype=np.float64)
    w = w / w.sum()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)

    oofs, tests = [], []
    for r in args.runs:
        o, t = load_run(Path(r))
        oofs.append(o); tests.append(t)
    truth = oofs[0][["true_viscosity", "true_oxidation"]].values.astype(np.float32)
    std_v = float(oofs[0]["true_viscosity"].std())
    std_o = float(oofs[0]["true_oxidation"].std())

    oof_stack = [o[["oof_viscosity", "oof_oxidation"]].values.astype(np.float32) for o in oofs]
    test_stack = [t[list(TEST_COLS)].values.astype(np.float32) for t in tests]
    test_ids = tests[0]["scenario_id"].values

    print("=== per-run OOF ===")
    for name, o in zip(args.runs, oof_stack):
        s, mv, mo = score(o, truth, std_v, std_o)
        print(f"  {name}: norm={s:.4f}  mv={mv:.3f}  mo={mo:.3f}")

    blended_oof = sum(wi * o for wi, o in zip(w, oof_stack))
    blended_test = sum(wi * t for wi, t in zip(w, test_stack))
    s, mv, mo = score(blended_oof, truth, std_v, std_o)
    print(f"\n=== blend with weights {dict(zip(args.runs, w.round(3)))} ===")
    print(f"  blended OOF norm={s:.4f}  mv={mv:.3f}  mo={mo:.3f}")

    pd.DataFrame({
        "scenario_id": oofs[0]["scenario_id"].values,
        "oof_viscosity": blended_oof[:, 0], "oof_oxidation": blended_oof[:, 1],
        "true_viscosity": truth[:, 0], "true_oxidation": truth[:, 1],
    }).to_csv(out_dir / "oof_predictions.csv", index=False)
    pd.DataFrame({
        "scenario_id": test_ids,
        TEST_COLS[0]: blended_test[:, 0],
        TEST_COLS[1]: blended_test[:, 1],
    }).to_csv(out_dir / "predictions_raw_headers.csv", index=False)
    with open(out_dir / "blend_weights.json", "w") as f:
        json.dump({"runs": list(args.runs),
                   "weights": [float(x) for x in w],
                   "oof_score": float(s)}, f, indent=2)
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
