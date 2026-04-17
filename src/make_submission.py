"""Pack inference.ipynb + predictions.csv into a zip for submission."""
import argparse
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="predictions.csv")
    ap.add_argument("--notebook", default="inference.ipynb")
    ap.add_argument("--out", default="submission.zip")
    args = ap.parse_args()

    files = [args.predictions, args.notebook]
    for f in files:
        if not Path(f).exists():
            raise SystemExit(f"Missing {f}")
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=Path(f).name)
    print(f"Wrote {args.out} with {files}")


if __name__ == "__main__":
    main()
