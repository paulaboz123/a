from typing import List, Dict, Any
import pandas as pd

def init():
    # Load your model here (e.g., transformers pipeline) if needed.
    pass

def run(mini_batch: List[str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for path in mini_batch:
        # Example expects CSV with a 'text' column. Adjust to your real input format.
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            text = str(row.get("text", ""))
            pred = "OK" if (len(text) % 2 == 0) else "NOK"
            results.append({"text": text, "prediction": pred})
    return results
