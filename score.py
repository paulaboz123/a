import json
from typing import Any, Dict, List, Union

def init():
    # Load your model here (transformers / torch).
    # Keep init fast and cache the model globally if needed.
    pass

def _iter_records_from_file(path: str):
    """Supports JSONL (one JSON per line) and JSON array."""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)

        # Heuristic: JSON array starts with '[' (ignoring whitespace)
        if head.lstrip().startswith('['):
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file must contain a list of objects.")
            for rec in data:
                yield rec
        else:
            # JSONL
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception as e:
                    yield {"_error": f"Invalid JSONL at line {line_no}: {e}", "_raw": line}

def run(mini_batch: List[str]) -> List[Dict[str, Any]]:
    """
    Azure ML Batch will call run(mini_batch) where mini_batch is a list of file paths.
    We return a list of dicts (rows) because deployment output_action=append_row.
    """
    results: List[Dict[str, Any]] = []

    for file_path in mini_batch:
        for rec in _iter_records_from_file(file_path):
            if isinstance(rec, dict) and rec.get("_error"):
                results.append({"source_file": file_path, "error": rec["_error"], "raw": rec.get("_raw")})
                continue

            # Your input schema: expects at least "text" (and optionally id, lang, etc.)
            text = ""
            rec_id = None
            if isinstance(rec, dict):
                rec_id = rec.get("id")
                text = rec.get("text", "")
            else:
                # If user sends plain strings in JSON array, support that too.
                text = str(rec)

            # TODO: Replace with your real model inference
            pred = "OK" if len(str(text)) % 2 == 0 else "NOK"

            results.append({
                "id": rec_id,
                "text": text,
                "prediction": pred
            })

    return results
