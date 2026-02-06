# score.py for Azure ML Batch endpoints
# - Keeps your input contract: JSON with keys: "document" and "num_preds"
# - Batch run() receives a list of file paths (mini-batch). Each file should contain JSON text
#   matching the same contract you used for online.
#
# Recommended input format for batch:
# - A folder (URI) containing *.json files, each one a single JSON request payload
#   OR a single .json file if you set mini_batch_size=1.
#
# Output:
# - For each input file, returns one dict. AML batch will materialize output as a file (depending on deployment settings).
#
import json
import os
import logging
import traceback
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Globals expected by your code
relevance_model = None
cd_logreg_model = None
cd_transformer_model = None


def init():
    """
    Called once per worker.
    """
    global relevance_model, cd_logreg_model, cd_transformer_model

    logger.info("Initializing model...")

    model_dir = os.getenv("AZUREML_MODEL_DIR")
    if not model_dir:
        raise RuntimeError("AZUREML_MODEL_DIR is not set")

    # Your code expects an "AL" subfolder; keep the same behavior.
    model_path = os.path.join(model_dir, "AL")

    logger.info(f"AZUREML_MODEL_DIR: {model_dir}")
    try:
        logger.info(f"Contents of AZUREML_MODEL_DIR: {os.listdir(model_dir)}")
    except Exception:
        pass

    logger.info(f"Using model path: {model_path}")
    try:
        logger.info(f"Contents of model path: {os.listdir(model_path)}")
    except Exception:
        pass

    # NOTE:
    # I am keeping your original loading calls/paths to avoid changing model packaging assumptions.
    # Replace these imports/classes with your real ones if needed.
    from models import LogisticRegression, Transformer  # your project modules

    relevance_model = LogisticRegression.load(os.path.join(model_path, "logreg_relevance.joblib"))
    cd_logreg_model = LogisticRegression.load(os.path.join(model_path, "logreg_cd.joblib"))

    transformer = os.path.join(model_path, "transformer_model")
    model_cd_transformer_path = os.path.join(transformer, "transformer")
    model_cd_transformer_le_path = os.path.join(transformer, "transformer_le.joblib")

    cd_transformer_model = Transformer.load(model_cd_transformer_path, model_cd_transformer_le_path)

    logger.info("Model initialized successfully.")


def _parse_request(raw: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Accepts either JSON string or dict.
    Returns validated dict with keys: document, num_preds.
    """
    if raw is None:
        raise ValueError("Bad Request: Request body cannot be empty!")

    if isinstance(raw, str):
        if raw.strip() == "":
            raise ValueError("Bad Request: Request body cannot be empty!")
        try:
            request_data = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError("Bad Request: Invalid JSON format!")
    elif isinstance(raw, dict):
        request_data = raw
    else:
        raise ValueError(f"Bad Request: Unsupported input type: {type(raw)}")

    if "document" not in request_data or "num_preds" not in request_data:
        raise ValueError("Bad Request: Invalid input, expected 'document' and 'num_preds'.")

    request_data["num_preds"] = int(request_data["num_preds"])
    return request_data


def is_relevant_customer_demand(
    text: str,
    relevant_proba: float,
    cd_logreg_proba: float,
    cd_transformer_proba: float,
) -> bool:
    # Keeping your logic from screenshot
    return (
        len(text.split(" ")) > 2
        and cd_logreg_proba > 0.1
        and ((relevant_proba > 0.65 and cd_transformer_proba > 0.9) or cd_transformer_proba > 0.95)
    )


def inference(document: Dict[str, Any], num_cd_predictions: int) -> Dict[str, Any]:
    import time
    import pandas as pd
    from pre_processing import clean_text  # your module

    # load and process document
    start_time = time.time()

    text = pd.Series(
        content["text"]
        for content in document["contentDomain"]["byId"].values()
    )

    text = clean_text(text)
    logger.info("Time to load and clean document: %s", time.time() - start_time)

    # score relevance model
    start_time = time.time()
    all_relevance_predictions = relevance_model.predict_proba(text)
    logger.info("Time to score relevance model: %s", time.time() - start_time)

    # score cd logreg model
    start_time = time.time()
    all_cd_logreg_predictions = cd_logreg_model.predict_top_n_labels_with_proba(text, num_cd_predictions)
    logger.info("Time to score cd logreg model: %s", time.time() - start_time)

    # score cd transformer
    start_time = time.time()
    all_cd_transformer_predictions = cd_transformer_model.predict_top_n_labels_with_proba(text, num_cd_predictions)
    logger.info("Time to score cd transformer model: %s", time.time() - start_time)

    # format results
    start_time = time.time()
    document_demand_predictions = set()

    for (
        content,
        relevance_prediction,
        cd_logreg_predictions,
        cd_transformer_predictions,
    ) in zip(
        document["contentDomain"]["byId"].values(),
        all_relevance_predictions,
        all_cd_logreg_predictions,
        all_cd_transformer_predictions,
    ):
        if is_relevant_customer_demand(
            content["text"],
            relevance_prediction,
            cd_logreg_predictions[0]["proba"],
            cd_transformer_predictions[0]["proba"],
        ):
            document_demand_predictions.add(cd_transformer_predictions[0]["label"])

        content.update(
            {
                "relevantProba": relevance_prediction,
                "cdLogregPredictions": cd_logreg_predictions,
                "cdTransformerPredictions": cd_transformer_predictions,
            }
        )

    document["documentDemandPredictions"] = list(document_demand_predictions)

    logger.info("Time to format results: %s", time.time() - start_time)
    return document


def run(mini_batch: List[str]) -> List[Dict[str, Any]]:
    """
    Batch entrypoint.
    mini_batch: list of file paths (strings) provided by Azure ML.
    Each file should contain JSON text with keys: document, num_preds.

    Returns a list of dict outputs (one per input).
    """
    outputs: List[Dict[str, Any]] = []

    for item in mini_batch:
        try:
            # Allow local/unit tests to pass raw JSON directly instead of a file path
            if isinstance(item, str) and os.path.exists(item):
                with open(item, "r", encoding="utf-8") as f:
                    raw_data = f.read()
            else:
                raw_data = item  # raw JSON string

            logger.info("Received item: %s", item if isinstance(item, str) else type(item))

            request_data = _parse_request(raw_data)
            document = request_data["document"]
            num_pred = request_data["num_preds"]

            response = inference(document, num_pred)

            # Keep a consistent wrapper like your online response did
            outputs.append({"predictions": response})

        except Exception as e:
            logger.error("Error processing item: %s", str(e))
            logger.error(traceback.format_exc())
            # For batch you usually DON'T want to crash whole job because one file is bad.
            outputs.append({"error": str(e)})

    return outputs
