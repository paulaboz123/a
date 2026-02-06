"""
Local smoke test for batch-style score.py.

Usage:
  python smoke_test_local.py --score-path ./score.py

It:
- builds a sample JSON input with keys: document + num_preds
- writes it to a temp file
- imports score.py as a module, calls init(), then run([path])
"""

import argparse
import importlib.util
import json
import os
import tempfile
from pathlib import Path


def import_module_from_path(path: str):
    spec = importlib.util.spec_from_file_location("score_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-path", required=True)
    args = ap.parse_args()

    score = import_module_from_path(args.score_path)

    # Minimal sample matching your code expectations:
    sample = {
        "document": {
            "contentDomain": {
                "byId": {
                    "1": {"text": "This is a sample customer demand text about pricing and delivery."},
                    "2": {"text": "Another text chunk with some request details."}
                }
            }
        },
        "num_preds": 3
    }

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "input.json"
        p.write_text(json.dumps(sample), encoding="utf-8")

        # For local tests you may not have AZUREML_MODEL_DIR or model files.
        # If you want to purely test parsing + batch wiring, comment out init().
        try:
            score.init()
        except Exception as e:
            print("init() failed (expected locally if models not present):", e)

        out = score.run([str(p)])
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
